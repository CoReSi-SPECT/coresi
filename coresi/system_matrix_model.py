# SPDX-FileCopyrightText: 2024 Vincent Lequertier <vincent@vl8r.eu>, Voichita Maxim <voichita.maxim@creatis.insa-lyon.fr>
# CREATIS Laboratory, INSA Lyon, France
# SPDX-License-Identifier: MIT

import os
import sys
from logging import getLogger
from math import pi

import numpy as np
import torch
import yaml

from coresi.camera import Camera, DetectorType
from coresi.event import Event
from coresi.image import Image
from coresi.interpolation import torch_1d_interp

logger = getLogger("CORESI")
_ = torch.set_grad_enabled(False)


class SM_Model(object):
    def __init__(
        self,
        config_mlem: dict,
        config_volume: dict,
        cameras: list[Camera],
        energies: list[int],
        tol: float,
    ):
        super(SM_Model, self).__init__()

        self.cone_thickness = config_mlem["cone_thickness"]

        self.compute_theta_j = True
        if config_mlem["model"] == "cos0rho0":
            # Theta_j is not needed for this model. Instruct to not compute it as
            # it's fairly expensive
            self.compute_theta_j = False
            self.do_nothing = True

            def model(kbl_j):
                return kbl_j

        elif config_mlem["model"] == "cos0rho2":
            # Theta_j is not needed for this model. Instruct to not compute it as
            # it's fairly expensive
            self.compute_theta_j = False
            self.do_nothing = False

            def model(rho_j):
                return 1 / rho_j**2

        elif config_mlem["model"] == "cos1rho2":
            self.do_nothing = False

            def model(cos_theta_j, rho_j):
                return abs(cos_theta_j) / rho_j**2

        else:
            logger.fatal(
                f"Model {config_mlem['model']} is not supported, either use cos0rho0, cos0rho2 or cos1rho2"
            )
            sys.exit(1)

        self.model = model
        self.cameras = cameras
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.m_e = torch.tensor(
            511, dtype=torch.float, device=self.device
        )  # electron mass in EV
        self.energies = torch.tensor(sorted(energies), device=self.device)
        self.n_energies = len(energies)
        self.tol = tol
        if self.n_energies == 0:
            logger.fatal("The configuration file has an empty E0 key")
            sys.exit(1)

        constants = self.read_constants("constants.yaml")
        logger.info(
            f"Using device {'cpu (' + str(os.cpu_count()) +' cpu available)' if not torch.cuda.is_available() else torch.cuda.get_device_name(0)}"
        )

        self.config_volume = config_volume
        self.line = Image(self.n_energies, self.config_volume)

        if self.cone_thickness == "parallel":
            self.sigma_beta = (
                self.line.voxel_size.norm2() * config_mlem["width_factor"] / 2
            )
            # Skip the Gaussian above n_sigma * Gaussian std
            self.limit_sigma = self.sigma_beta * config_mlem["n_sigma"]
            if self.n_energies > 1 or config_mlem["force_spectral"]:
                # Alias to avoid selecting the right algorithm in the run loop
                self.SM_line = self.SM_parallel_thickness_spectral
            else:
                self.SM_line = self.SM_parallel_thickness
        elif self.cone_thickness in ["angular", "angular_precise"]:
            # TODO: If E0 is unknown, we need to interpolate the values for each
            # event based on the known constants
            # TODO: What we don't have the constants for a given energy?
            # Spectral or not. Same. interpolate
            # We need constants for each energy. Angular resolution modeling
            # is a double Gaussian, its parameters depend on the energy
            self.a1, self.a2, self.sigma_beta_1, self.sigma_beta_2 = [], [], [], []
            for energy in self.energies:
                key = f"energy_{energy}"
                if key in constants["doppler_broadening"]:
                    self.a1.append(constants["doppler_broadening"][key]["a1"])
                    self.a2.append(constants["doppler_broadening"][key]["a2"])
                    self.sigma_beta_1.append(
                        constants["doppler_broadening"][key]["sigma_beta_1"]
                        * config_mlem["width_factor"]
                    )
                    self.sigma_beta_2.append(
                        constants["doppler_broadening"][key]["sigma_beta_2"]
                        * config_mlem["width_factor"]
                    )
                else:
                    # interpolate
                    # TODO: sort values according to the right key
                    known_energies = torch.tensor(
                        [
                            int(key.split("_")[1])
                            for key in constants["doppler_broadening"].keys()
                        ],
                        device=self.device,
                    )
                    self.a1.append(
                        torch_1d_interp(
                            energy,
                            known_energies,
                            torch.tensor(
                                [
                                    value["a1"]
                                    for value in constants[
                                        "doppler_broadening"
                                    ].values()
                                ],
                                device=self.device,
                            ),
                        )
                    )
                    self.a2.append(
                        torch_1d_interp(
                            energy,
                            known_energies,
                            torch.tensor(
                                [
                                    value["a2"]
                                    for value in constants[
                                        "doppler_broadening"
                                    ].values()
                                ],
                                device=self.device,
                            ),
                        )
                    )
                    self.sigma_beta_1.append(
                        config_mlem["width_factor"]
                        * torch_1d_interp(
                            energy,
                            known_energies,
                            torch.tensor(
                                [
                                    value["sigma_beta_1"]
                                    for value in constants[
                                        "doppler_broadening"
                                    ].values()
                                ],
                                device=self.device,
                            ),
                        )
                    )
                    self.sigma_beta_2.append(
                        config_mlem["width_factor"]
                        * torch_1d_interp(
                            energy,
                            known_energies,
                            torch.tensor(
                                [
                                    value["sigma_beta_2"]
                                    for value in constants[
                                        "doppler_broadening"
                                    ].values()
                                ],
                                device=self.device,
                            ),
                        )
                    )
                self.limit_sigma = [
                    max([self.sigma_beta_1[idx], self.sigma_beta_2[idx]])
                    * config_mlem["n_sigma"]
                    for idx in range(len(self.sigma_beta_1))
                ]
            if self.n_energies == 1 and not config_mlem["force_spectral"]:
                # If only one energy, tranform the array into a single value
                self.a1 = self.a1[0]
                self.a2 = self.a2[0]
                self.sigma_beta_1 = self.sigma_beta_1[0]
                self.sigma_beta_2 = self.sigma_beta_2[0]
                self.limit_sigma = (
                    max([self.sigma_beta_1, self.sigma_beta_2]) * config_mlem["n_sigma"]
                )
                self.SM_line = self.SM_angular_thickness
            elif self.cone_thickness == "angular":
                self.SM_line = self.SM_angular_thickness_spectral
            elif self.cone_thickness == "angular_precise":
                self.SM_line = self.SM_angular_thickness_spectral_precise

        logger.info(f"Using algorithm {self.SM_line.__name__}")

        x, y, z = SM_Model.create_mesh_axes(
            [
                self.line.corner.x + (self.line.voxel_size.x / 2),
                self.line.corner.x
                + self.line.dim_in_cm.x
                - (self.line.voxel_size.x / 2),
            ],
            self.line.dim_in_voxels.x,
            [
                self.line.corner.y + (self.line.voxel_size.y / 2),
                self.line.corner.y
                + self.line.dim_in_cm.y
                - (self.line.voxel_size.y / 2),
            ],
            self.line.dim_in_voxels.y,
            [
                self.line.corner.z + (self.line.voxel_size.z / 2),
                self.line.corner.z
                + self.line.dim_in_cm.z
                - (self.line.voxel_size.z / 2),
            ],
            self.line.dim_in_voxels.z,
        )

        # Used to go through the volume
        self.xx, self.yy, self.zz = np.meshgrid(x, y, z, sparse=True, indexing="ij")
        self.xx = torch.from_numpy(self.xx).to(self.device)
        self.yy = torch.from_numpy(self.yy).to(self.device)
        self.zz = torch.from_numpy(self.zz).to(self.device)

    def SM_angular_thickness(
        self, event: Event, check_valid_events: bool = True
    ) -> Image:
        # rho_j is a vector with distances from the voxel to the cone origin
        # It's normalized
        rho_j = torch.sqrt(
            (event.V1.x - self.xx) ** 2
            + (event.V1.y - self.yy) ** 2
            + (event.V1.z - self.zz) ** 2
        ).unsqueeze(0)
        self.line.set_to_zeros()

        # delta j is  angle from the cone axis to the voxel
        # self.xx - event.V1.x is Oj v1. We need it further down in the code but it
        # might take a lot of memory to store it. So compute it inline here and
        # below for theta_j. If in the future we have a lot of ram it can be
        # stored
        cos_delta_j = (
            event.axis.x * (self.xx - event.V1.x)
            + event.axis.y * (self.yy - event.V1.y)
            + event.axis.z * (self.zz - event.V1.z)
        ) / rho_j
        # Because of floating point arithmetics, the cosinus could outside the
        # -1,1 range. Clamp the values to workaround that.
        cos_delta_j = torch.clamp(cos_delta_j, min=-1.0, max=1.0)

        # ddelta is the angle from the voxels to the cone surface
        self.line.values = torch.abs(torch.arccos(cos_delta_j) - event.beta)

        # Discard voxels not within the "thick" cone
        mask = self.line.values > self.limit_sigma
        self.line.values[mask] = 0.0

        # If the cone does not intersect with the volume at all, discard the self.line
        if check_valid_events and not torch.any(self.line.values):
            raise ValueError(
                f"The cone does not intersect the volume for event {event.id}"
            )
        # Remove the background for cos_delta_j
        cos_delta_j[mask] = 0.0
        camera_V1_Oz = self.cameras[event.idx_V1].Oz
        # move further away from the cone boundary
        # Do not compute gaussian for the background outside the cone
        mask = ~mask
        # Apply the Gaussian
        self.line.values[mask] = self.a1 * torch.exp(
            -(self.line.values[mask] ** 2) * 0.5 / self.sigma_beta_1**2
        ) + self.a2 * torch.exp(
            -(self.line.values[mask] ** 2) * 0.5 / self.sigma_beta_2**2
        )

        # lambda / lambda prime
        # TODO: use get_compton_diff_xsection
        KN = 1.0 / (1.0 + event.E0 / self.m_e * (1.0 - cos_delta_j[mask]))

        # KN is the Klein–Nishina formula to obtain the differential cross
        # section for each voxel. https://en.wikipedia.org/wiki/Klein%E2%80%93Nishina_formula
        # sin**2 = 1-cos**2 to avoid computing the sinus
        # Note that the coresi C++ did not use the cos squared, this might have
        # been a mistake
        KN = KN * (KN**2 + 1) + KN * KN * (-1.0 + cos_delta_j[mask] ** 2)

        # Compute the angle between the camera z axis and the voxel
        # "solid angle". theta_j
        if self.compute_theta_j:
            # TODO: Apply mask here?
            cos_theta_j = (
                (camera_V1_Oz.x * (self.xx - event.V1.x))
                + (camera_V1_Oz.y * (self.yy - event.V1.y))
                + (camera_V1_Oz.z * (self.zz - event.V1.z)) / rho_j
            )
            self.line.values *= self.model(cos_theta_j, rho_j)
        elif not self.do_nothing:
            self.line.values *= self.model(rho_j)

        # Ti is here i.e. the system matrix for an event i
        self.line.values[mask] = KN * self.line.values[mask]

        return self.line

    def SM_angular_thickness_spectral(
        self, event: Event, check_valid_events: bool = True
    ) -> Image:
        """docstring for SM_parallel_thickness_spectral"""
        if event.energy_bin >= self.n_energies:
            logger.fatal(
                f"The energy bin has not been determinted correctly for event {str(event.id)}"
            )
            sys.exit(1)
        camera = self.cameras[event.idx_V1]
        self.line.set_to_zeros()

        # rho_j is a vector with distances from the voxel to the cone origin
        # It's normalized
        rho_j = torch.sqrt(
            (event.V1.x - self.xx) ** 2
            + (event.V1.y - self.yy) ** 2
            + (event.V1.z - self.zz) ** 2
        )
        if self.compute_theta_j:
            # We use the meshgrid twice to avoid storing the result because it's 3 times the
            # volume size
            cos_theta_j = (
                (event.V1.x - self.xx) * event.normal_to_layer_V1.x
                + (event.V1.y - self.yy) * event.normal_to_layer_V1.y
                + (event.V1.z - self.zz) * event.normal_to_layer_V1.z
            ) / rho_j

        # delta j is the angle from the cone axis to the voxel
        # self.xx - event.V1.x is Oj v1. We need it further down in the code but it
        # might take a lot of memory to store it. So compute it inline here and
        # below for theta_j. If in the future we have a lot of ram it can be
        # stored
        cos_delta_j = (
            event.axis.x * (self.xx - event.V1.x)
            + event.axis.y * (self.yy - event.V1.y)
            + event.axis.z * (self.zz - event.V1.z)
        ) / rho_j

        # Because of floating point arithmetics, the cosinus could outside the
        # -1,1 range. Clamp the values to workaround that.
        cos_delta_j = torch.clamp(cos_delta_j, min=-1.0, max=1.0)

        # Geometry
        for idx in torch.where(event.xsection > 0.0)[0]:
            cos_beta = 1.0 - (
                self.m_e
                * event.Ee
                / (self.energies[idx] * (self.energies[idx] - event.Ee))
            )

            if check_valid_events and ((cos_beta < -1) or (cos_beta > 1)):
                event.xsection[idx] = 0.0
                continue
            # We take the sinus (optimized) of ddelta (angle from the voxels to the cone surface)
            # and multiply by rhoj to get the distance from
            # the voxels to the cone surface
            self.line.values[idx] = torch.abs(
                torch.arccos(cos_delta_j) - torch.arccos(cos_beta)
            )
            mask_cone = self.line.values[idx] <= self.limit_sigma[idx]

            self.line.values[idx][~mask_cone] = 0.0
            # If the cone does not intersect the volume for a given energy,
            # continue
            if check_valid_events and torch.all(~mask_cone):
                event.xsection[idx] = 0.0
                continue

            # Gauss
            # TODO: Was the a1 values computed by regression with the div by (2
            # * self.sigma_beta_1[idx] ** 2)?
            self.line.values[idx][mask_cone] = self.a1[idx] * torch.exp(
                -(self.line.values[idx][mask_cone] ** 2)
                / (2 * self.sigma_beta_1[idx] ** 2)
            ) + self.a2[idx] * torch.exp(
                -(self.line.values[idx][mask_cone] ** 2)
                / (2 * self.sigma_beta_2[idx] ** 2)
            )

            if self.compute_theta_j:
                self.line.values[idx] *= self.model(cos_theta_j, rho_j)
            elif not self.do_nothing:
                self.line.values[idx] *= self.model(rho_j)

            # Physics
            sca_compton_diff_xsection = camera.get_compton_diff_xsection(
                self.energies[idx],
                # ENRIQUE: In Enrique thesis the cosbeta here is
                # the one of the known energy constant
                # for all voxels. Instead we use
                # cos_delta_j which is a matrix
                cos_beta,
            )

            int2Xsect = 0.0
            photo_x_section = camera.get_photo_diff_xsection(
                self.energies[idx] - event.Ee, DetectorType.ABS
            )
            photo_x_section_m_e = camera.get_photo_diff_xsection(
                self.m_e, DetectorType.ABS
            )

            pair_x_section = camera.get_pair_diff_xsection(
                self.energies[idx] - event.Ee, DetectorType.ABS
            )
            # Absorption total is Photoelectric, partial absorption is either
            # Compton or pair production
            # Photoelectric
            if abs(self.energies[idx] - event.E0) < self.tol:
                int2Xsect = (
                    camera.abs_density * photo_x_section
                    # Include double absorption probability after pair creation
                    + camera.abs_density
                    * pair_x_section
                    * 2
                    * self.tol
                    * torch.pow(
                        (
                            1
                            - torch.exp(
                                -photo_x_section_m_e
                                * camera.abs_density
                                * (camera.sca_layers + camera.abs_layers)[
                                    event.layer_idx_V2
                                ].thickness
                                / 2.0
                            )
                        ),
                        2,
                    )
                )
            # Compton allowed
            else:
                cos_beta_2 = 1.0 - (
                    self.m_e
                    * event.Eg
                    / (
                        (self.energies[idx] - event.Ee)
                        * (self.energies[idx] - event.Ee - event.Eg)
                    )
                )
                if abs(cos_beta_2) <= 1.0:
                    abs_compton_diff_xsection = camera.get_compton_diff_xsection(
                        self.energies[idx],
                        # ENRIQUE: In Enrique thesis the cosbeta here is
                        # the one of the known energy constant
                        # for all voxels. Instead we use
                        # cos_delta_j which is a matrix
                        cos_beta_2,
                    )

                    int2Xsect = (
                        camera.abs_n_eff
                        * abs_compton_diff_xsection
                        * self.m_e
                        * pi
                        * 4
                        # dE = tol * 2
                        # ENRIQUE: dE is not documented in the original C++ code
                        * self.tol
                        / (torch.pow(self.energies[idx] - event.E0, 2))
                    )

                # Test for pair production
                if (
                    abs(self.energies[idx] - (event.E0 + (2 * self.m_e))) < 2 * self.tol
                ):  # with double escape
                    int2Xsect += (
                        camera.abs_density
                        * pair_x_section
                        * torch.exp(
                            -photo_x_section_m_e
                            * camera.abs_density
                            * (camera.sca_layers + camera.abs_layers)[
                                event.layer_idx_V2
                            ].thickness
                        )
                    )

                elif (
                    abs(self.energies[idx] - (event.E0 + self.m_e)) < 2 * self.tol
                ):  # with single escape
                    int2Xsect += (
                        2
                        * camera.abs_density
                        * pair_x_section
                        * (
                            torch.exp(
                                -photo_x_section_m_e
                                * camera.abs_density
                                * (camera.sca_layers + camera.abs_layers)[
                                    event.layer_idx_V2
                                ].thickness
                                / 2.0
                            )
                            # ENRIQUE: why difference of two exponentials. Why not
                            # multiplication of probabilities?. and exp * (1-exp)?
                            - torch.exp(
                                --photo_x_section_m_e
                                * camera.abs_density
                                * (camera.sca_layers + camera.abs_layers)[
                                    event.layer_idx_V2
                                ].thickness
                            )
                        )
                    )
            self.line.values[idx][mask_cone] = (
                int2Xsect * self.line.values[idx][mask_cone]
            )
            if int2Xsect == 0.0:
                event.xsection[idx] = 0.0
                continue

            kbl_j = (
                camera.sca_n_eff
                * sca_compton_diff_xsection
                * self.m_e
                * 2
                * torch.pi
                # What Enrique did - use the E1 from the event
                / torch.pow(self.energies[idx] - event.Ee, 2)
            )

            # Multiply by exponential terms
            # See List mode em reconstruction of Compton Sctter Camera Images in 3D.
            # Wilderman et al. for help and Enrique's thesis.
            # Probabilities for the first photon to reach the middle of the
            # scatterer layer
            kbl_j *= torch.exp(
                # zd11
                -camera.get_total_diff_xsection(
                    self.energies[idx], event.detector_type_V1
                )
                # Attenuation coefficient is proportinal to the density
                # (which depends on the medium's physical state)
                * (
                    camera.sca_density
                    if event.detector_type_V1 == DetectorType.SCA
                    else camera.abs_density
                )
                # The thickness. layer_idx_V1 is in a defined layer i.e. it
                # will not be in a None absortber layer as this is checked
                # beforehand
                * (camera.sca_layers + camera.abs_layers)[event.layer_idx_V1].thickness
                # Assume the interaction is the middle of the material in a
                # orthogonal line?
                # TODO: take the incident angle into account?
                / 2
            )

            # zd12
            # Probability of going out of the scatterer layer and to the
            # absorber
            kbl_j *= torch.exp(
                -camera.get_total_diff_xsection(
                    self.energies[idx] - event.Ee, event.detector_type_V1
                )
                * (
                    camera.sca_density
                    if event.detector_type_V1 == DetectorType.SCA
                    else camera.abs_density
                )
                * (camera.sca_layers + camera.abs_layers)[event.layer_idx_V1].thickness
                / 2
            )
            # Attenuation in planes not triggered
            # Before scatterer
            # TODO: If first interaction can be in the absorber. In this case, we
            # would need to compute the number of scatteres the gamma went
            # through without interactions
            # Probability of first and 2nd photon to go through not triggered
            # layers, respectivelly
            if event.layer_idx_V1 > 0:
                kbl_j *= torch.exp(
                    -(
                        camera.get_total_diff_xsection(
                            self.energies[idx], event.detector_type_V1
                        )
                        * (
                            camera.sca_density
                            if event.detector_type_V1 == DetectorType.SCA
                            else camera.abs_density
                        )
                        * event.layer_idx_V1
                        * (camera.sca_layers + camera.abs_layers)[
                            event.layer_idx_V1
                        ].thickness
                    )
                )
            # After scatterer
            # TODO: This works only for 1 single absorber below the scatterers
            # But what if the interaction is in an absorber on the sides. If
            # that happens, the scatterers below may needs to be ignored
            kbl_j *= torch.exp(
                -(
                    camera.get_total_diff_xsection(
                        self.energies[idx] - event.Ee, event.detector_type_V1
                    )
                    * (
                        camera.sca_density
                        if event.detector_type_V1 == DetectorType.SCA
                        else camera.abs_density
                    )
                    * abs(event.layer_idx_V2 - event.layer_idx_V1 - 1)
                    * (camera.sca_layers + camera.abs_layers)[
                        event.layer_idx_V1
                    ].thickness
                )
            )

            self.line.values[idx][mask_cone] = kbl_j * self.line.values[idx][mask_cone]

        if check_valid_events and torch.all(event.xsection == 0.0):
            raise ValueError(
                f"The cone does not intersect the volume for event {event.id} for all tried energies"
            )

        return self.line

    def SM_angular_thickness_spectral_precise(
        self, event: Event, check_valid_events: bool = True
    ) -> Image:
        if event.energy_bin >= self.n_energies:
            logger.fatal(
                f"The energy bin has not been determinted correctly for event {str(event.id)}"
            )
            sys.exit(1)
        self.line.set_to_zeros()
        camera = self.cameras[event.idx_V1]

        # rho_j is a vector with distances from the voxels to the cone origin
        # It's normalized
        rho_j = torch.sqrt(
            (event.V1.x - self.xx) ** 2
            + (event.V1.y - self.yy) ** 2
            + (event.V1.z - self.zz) ** 2
        )
        # We use the meshgrid twice to avoid storing the difference because it's 3 times the
        # volume size
        cos_theta_j = (
            (event.V1.x - self.xx) * event.normal_to_layer_V1.x
            + (event.V1.y - self.yy) * event.normal_to_layer_V1.y
            + (event.V1.z - self.zz) * event.normal_to_layer_V1.z
        ) / rho_j

        # delta j is the angle from the cone axis to the voxels
        # self.xx - event.V1.x is Oj v1. We need it further down in the code but it
        # might take a lot of memory to store it. So compute it inline here and
        # below for theta_j. If in the future we have a lot of ram it can be
        # stored
        cos_delta_j = (
            event.axis.x * (self.xx - event.V1.x)
            + event.axis.y * (self.yy - event.V1.y)
            + event.axis.z * (self.zz - event.V1.z)
        ) / rho_j

        # Because of floating point arithmetics, the cosinus could outside the
        # -1,1 range. Clamp the values to workaround that.
        cos_delta_j = torch.clamp(cos_delta_j, min=-1.0, max=1.0)

        x_section_m_e = camera.get_photo_diff_xsection(self.m_e, DetectorType.ABS)

        # Geometry
        for idx in torch.where(event.xsection > 0.0)[0]:
            cos_beta = 1.0 - (
                self.m_e
                * event.Ee
                / (self.energies[idx] * (self.energies[idx] - event.Ee))
            )

            if check_valid_events and ((cos_beta < -1) or (cos_beta > 1)):
                event.xsection[idx] = 0.0
                continue

            # ddelta is the angle from the voxels to the cone surface
            self.line.values[idx] = torch.abs(
                torch.arccos(cos_delta_j) - torch.arccos(cos_beta)
            )
            mask_cone = self.line.values[idx] <= self.limit_sigma[idx]

            self.line.values[idx][~mask_cone] = 0.0
            # If the cone does not intersect the volume for a given energy,
            # continue
            if check_valid_events and torch.all(~mask_cone):
                event.xsection[idx] = 0.0
                continue

            # Gauss
            self.line.values[idx][mask_cone] = self.a1[idx] * torch.exp(
                -(self.line.values[idx][mask_cone] ** 2)
                / (2 * self.sigma_beta_1[idx] ** 2)
            ) + self.a2[idx] * torch.exp(
                -(self.line.values[idx][mask_cone] ** 2)
                / (2 * self.sigma_beta_2[idx] ** 2)
            )

            if self.compute_theta_j:
                self.line.values[idx] *= self.model(cos_theta_j, rho_j)
            elif not self.do_nothing:
                self.line.values[idx] *= self.model(rho_j)

            # Recompute the energy of the scattered photon to account for the
            # assumed total energy provided by the configuration
            E_gamma = self.energies[idx] / (
                1 + (self.energies[idx] / self.m_e) * (1 - cos_delta_j[mask_cone])
            )
            density_V1 = (
                camera.sca_density
                if event.detector_type_V1 == DetectorType.SCA
                else camera.abs_density
            )
            density_V2 = (
                camera.abs_density
                if event.detector_type_V2 == DetectorType.ABS
                else camera.sca_density
            )

            sca_compton_diff_xsection = camera.get_compton_diff_xsection(
                self.energies[idx],
                # ENRIQUE: In Enrique thesis the cosbeta here is
                # the one of the known energy constant
                # for all voxels. Instead we use
                # cos_delta_j which is a matrix
                cos_delta_j[mask_cone],
            )

            int2Xsect = torch.zeros_like(E_gamma)
            # Total absorbtion is Photoelectric, partial absorbition is either
            # compton or pair production
            # Photoelectric
            mask_tot = torch.abs(E_gamma - event.Eg) < self.tol
            # Create a generic x_section variable for different physical
            # effects. Reused to avoid creating new volumes and optimize
            # memory
            if torch.any(mask_tot):
                x_section = camera.get_photo_diff_xsection(
                    E_gamma[mask_tot], event.detector_type_V2
                )
                int2Xsect[mask_tot] = camera.abs_density * x_section * 2 * self.tol
                x_section = camera.get_pair_diff_xsection(
                    E_gamma[mask_tot], event.detector_type_V2
                )
                # Include double absorption probability after pair creation in
                # the middle of the absorber
                double_abs = torch.pow(
                    1.0
                    - torch.exp(
                        -x_section_m_e
                        * density_V2
                        * (camera.sca_layers + camera.abs_layers)[
                            event.layer_idx_V2
                        ].thickness
                        / 2.0
                    ),
                    2,
                )
                int2Xsect[mask_tot] += double_abs * camera.abs_density * x_section
            # 2nd Compton scattering
            cos_beta_2 = 1.0 - (
                # E_gamma - event.Eg is the energy after the 2nd Compton
                # scaterring
                self.m_e
                * event.Eg
                / (E_gamma * (E_gamma - event.Eg))
            )
            mask_partial = ~mask_tot & (torch.abs(cos_beta_2) <= 1.0)
            if torch.any(mask_partial):
                abs_compton_diff_xsection = camera.get_compton_diff_xsection(
                    self.energies[idx],
                    cos_beta_2[mask_partial],
                )

                int2Xsect[mask_partial] = (
                    camera.abs_n_eff
                    * abs_compton_diff_xsection
                    * self.m_e
                    * pi
                    * 4
                    # dE = tol * 2
                    # ENRIQUE: dE is not documented in the original C++ code
                    * self.tol
                    / (torch.pow(E_gamma[mask_partial] - event.Eg, 2))
                )

            mask_partial = ~mask_tot & (
                torch.abs(E_gamma - event.Eg - 2 * self.m_e) < 2 * self.tol
            )
            if torch.any(mask_partial):
                x_section = camera.get_pair_diff_xsection(
                    E_gamma[mask_partial], event.detector_type_V2
                )
                int2Xsect[mask_partial] += (
                    density_V2
                    * x_section
                    * torch.exp(
                        -x_section_m_e
                        * density_V2
                        * (camera.sca_layers + camera.abs_layers)[
                            event.layer_idx_V2
                        ].thickness
                    )
                )
            mask_pair_single = (
                ~mask_tot
                & ~mask_partial
                & (torch.abs(E_gamma - event.Eg - self.m_e) < 2 * self.tol)
            )

            if torch.any(mask_pair_single):
                x_section = camera.get_pair_diff_xsection(
                    E_gamma[mask_pair_single], event.detector_type_V2
                )
                int2Xsect[mask_pair_single] += (
                    2
                    * density_V2
                    * x_section
                    * (
                        torch.exp(
                            -x_section_m_e
                            * density_V2
                            * (camera.sca_layers + camera.abs_layers)[
                                event.layer_idx_V2
                            ].thickness
                            / 2.0
                        )
                        - torch.exp(
                            -x_section_m_e
                            * density_V2
                            * (camera.sca_layers + camera.abs_layers)[
                                event.layer_idx_V2
                            ].thickness
                        )
                    )
                )
            del mask_tot, mask_partial, mask_pair_single
            self.line.values[idx][mask_cone] = (
                self.line.values[idx][mask_cone] * int2Xsect
            )
            if torch.all(int2Xsect == 0.0):
                event.xsection[idx] = 0.0
                continue

            # ENRIQUE: Different from Enrique's thesis: no V2V1^2 is considered here, the
            # constants, and the probability of escape of the secondary photon (i.e. no interaction)
            # second interaction
            # First interaction is here
            prob_interactions = (
                camera.sca_n_eff
                * sca_compton_diff_xsection
                * self.m_e
                * 4
                * pi
                / torch.pow(E_gamma, 2)
            )

            # Multiply by exponential terms
            # See List mode em reconstruction of Compton Sctter Camera Images in 3D.
            # Wilderman et al. for help and Enrique's thesis.
            # Probabilities for the first photon to reach the middle of the
            # scatterer layer
            prob_interactions *= torch.exp(
                # zd11
                -camera.get_total_diff_xsection(
                    self.energies[idx], event.detector_type_V1
                )
                # Attenuation coefficient is proportinal to the density
                # (which depends on the medium's physical state)
                * density_V1
                # The thickness. layer_idx_V1 is in a defined layer i.e. it
                # will not be in a None absortber layer as this is checked
                # beforehand
                * (camera.sca_layers + camera.abs_layers)[event.layer_idx_V1].thickness
                # Assume the interaction is the middle of the material in a
                # orthogonal line?
                # TODO: take the incident angle into account?
                / 2
            )

            # zd12
            # Probability of of going out of the scatterer layer and to the
            # absorber
            prob_interactions *= torch.exp(
                -camera.get_total_diff_xsection(E_gamma, event.detector_type_V1)
                * density_V1
                * (camera.sca_layers + camera.abs_layers)[event.layer_idx_V1].thickness
                / 2
            ) * torch.exp(
                -(
                    camera.get_total_diff_xsection(E_gamma, event.detector_type_V2)
                    * density_V2
                    * (camera.sca_layers + camera.abs_layers)[
                        event.layer_idx_V2
                    ].thickness
                    / 2
                )
            )
            # Attenuation in planes not triggered
            # Before scatterer
            # TODO: If first interaction can be in the absorber. In this case, we
            # would need to compute the number of scatteres the gamma went
            # through without interactions
            # Probability of first and 2nd photon to go through not triggered
            # layers, respectivelly
            if event.layer_idx_V1 > 0:
                prob_interactions *= torch.exp(
                    -(
                        camera.get_total_diff_xsection(
                            self.energies[idx], event.detector_type_V1
                        )
                        * density_V1
                        * event.layer_idx_V1
                        * (camera.sca_layers + camera.abs_layers)[
                            event.layer_idx_V1
                        ].thickness
                    )
                )
            # After scatterer
            # TODO: This works only for 1 single absorber below the scatterers
            # But what if the interaction is in an absorber on the sides. If
            # that happens, the scatterers below may needs to be ignored
            prob_interactions *= torch.exp(
                -(
                    camera.get_total_diff_xsection(E_gamma, event.detector_type_V1)
                    * density_V1
                    * abs(event.layer_idx_V2 - event.layer_idx_V1 - 1)
                    * (camera.sca_layers + camera.abs_layers)[
                        event.layer_idx_V1
                    ].thickness
                )
            )

            self.line.values[idx][mask_cone] = (
                prob_interactions * self.line.values[idx][mask_cone]
            )

        if check_valid_events and torch.all(event.xsection == 0.0):
            raise ValueError(
                f"The cone does not intersect the volume for event {event.id} for all tried energies"
            )

        return self.line

    def SM_parallel_thickness(
        self, event: Event, check_valid_events: bool = True
    ) -> Image:
        """docstring for SM_parallel_thickness"""
        # rho_j is a vector with distances from the voxel to the cone origin
        # It's normalized
        rho_j = torch.sqrt(
            (event.V1.x - self.xx) ** 2
            + (event.V1.y - self.yy) ** 2
            + (event.V1.z - self.zz) ** 2
        ).unsqueeze(dim=0)
        # delta j is the distance from the voxel to the cone axis
        # self.xx - event.V1.x is Oj v1. We need it further down in the code but it
        # might take a lot of memory to store it. So compute it inline here and
        # below for theta_j. If in the future we have a lot of ram it can be
        # stored
        # Cosinus of delta_j
        cos_delta_j = (
            event.axis.x * (self.xx - event.V1.x)
            + event.axis.y * (self.yy - event.V1.y)
            + event.axis.z * (self.zz - event.V1.z)
        ) / rho_j
        # Because of floating point arithmetics, the cosinus could outside the
        # -1,1 range. Clamp the values to workaround that.
        cos_delta_j = torch.clamp(cos_delta_j, min=-1.0, max=1.0)

        # We take the sinus (optimized) of ddelta (angle from the voxels to the cone surface)
        # and multiply by rhoj to get the distance from
        # the voxels to the cone surface
        self.line.values = rho_j * torch.abs(
            event.cosbeta * torch.sqrt(1 - cos_delta_j**2) - event.sinbeta * cos_delta_j
        )
        # Discard voxels not within the "thick" cone
        mask = self.line.values > self.limit_sigma
        self.line.values[mask] = 0.0

        # If the cone does not intersect with the voxel at all, discard the self.line
        if check_valid_events and not torch.any(self.line.values):
            raise ValueError(
                f"The cone does not intersect the voxel for event {event.id}"
            )
        # Remove the background for cos_delta_j
        cos_delta_j[mask] = 0.0
        camera_V1_Oz = self.cameras[event.idx_V1].Oz
        # move further away from the cone boundary
        # Do not compute gaussian for the background outside the cone
        mask = ~mask
        # Apply the Gaussian
        self.line.values[mask] = torch.exp(
            -(self.line.values[mask] ** 2) * 0.5 / self.sigma_beta**2
        )

        if self.compute_theta_j:
            cos_theta_j = (
                (camera_V1_Oz.x * (self.xx - event.V1.x))
                + (camera_V1_Oz.y * (self.yy - event.V1.y))
                + (camera_V1_Oz.z * (self.zz - event.V1.z)) / rho_j
            )
            self.line.values *= self.model(cos_theta_j, rho_j)
        elif not self.do_nothing:
            self.line.values *= self.model(rho_j)

        # lambda / lambda prime
        # TODO: use get_compton_diff_xsection
        KN = 1.0 / (1.0 + event.E0 / self.m_e * (1.0 - cos_delta_j[mask]))

        # KN is the Klein–Nishina formula to obtain the differential cross
        # section for each voxel. https://en.wikipedia.org/wiki/Klein%E2%80%93Nishina_formula
        # sin**2 = 1-cos**2 to avoid computing the sinus
        # Note that the coresi C++ did not use the cos squared, this might have
        # been a mistake
        KN = KN * (KN**2 + 1) + KN * KN * (-1.0 + cos_delta_j[mask] ** 2)

        # Ti is here i.e. the system matrix for an event i
        self.line.values[mask] = KN * self.line.values[mask]
        return self.line

    def SM_parallel_thickness_spectral(
        self, event: Event, check_valid_events: bool = True
    ) -> Image:
        """docstring for SM_parallel_thickness_spectral"""
        if event.energy_bin >= self.n_energies:
            logger.fatal(
                f"The energy bin has not been determinted correctly for event {str(event.id)}"
            )
            sys.exit(1)
        self.line.set_to_zeros()
        camera = self.cameras[event.idx_V1]
        # self.line.values = torch.zeros(self.line.values.shape)

        # rho_j is a vector with distances from the voxel to the cone origin
        # It's normalized
        rho_j = torch.sqrt(
            (event.V1.x - self.xx) ** 2
            + (event.V1.y - self.yy) ** 2
            + (event.V1.z - self.zz) ** 2
        )
        if self.compute_theta_j:
            # We use the meshgrid twice to avoid storing the result because it's 3 times the
            # volume size
            cos_theta_j = (
                (event.V1.x - self.xx) * event.normal_to_layer_V1.x
                + (event.V1.y - self.yy) * event.normal_to_layer_V1.y
                + (event.V1.z - self.zz) * event.normal_to_layer_V1.z
            ) / rho_j

        # delta j is the angle from the cone axis to the voxel
        # self.xx - event.V1.x is Oj v1. We need it further down in the code but it
        # might take a lot of memory to store it. So compute it inline here and
        # below for theta_j. If in the future we have a lot of ram it can be
        # stored
        cos_delta_j = (
            event.axis.x * (self.xx - event.V1.x)
            + event.axis.y * (self.yy - event.V1.y)
            + event.axis.z * (self.zz - event.V1.z)
        ) / rho_j
        # Because of floating point arithmetics, the cosinus could outside the
        # -1,1 range. Clamp the values to workaround that.
        cos_delta_j = torch.clamp(cos_delta_j, min=-1.0, max=1.0)

        # Geometry
        for idx in torch.where(event.xsection > 0.0)[0]:
            cos_beta = 1.0 - (
                self.m_e
                * event.Ee
                / (self.energies[idx] * (self.energies[idx] - event.Ee))
            )

            if check_valid_events and ((cos_beta < -1) or (cos_beta > 1)):
                event.xsection[idx] = 0.0
                continue
            sin_beta = torch.sqrt(1 - torch.pow(cos_beta, 2))
            # We take the sinus (optimized) of ddelta (angle from the voxels to the cone surface)
            # and multiply by rhoj to get the distance from
            # the voxels to the cone surface
            self.line.values[idx] = rho_j * torch.abs(
                cos_beta * torch.sqrt(1 - cos_delta_j**2) - sin_beta * cos_delta_j
            )
            mask_cone = self.line.values[idx] <= self.limit_sigma

            self.line.values[idx][~mask_cone] = 0.0
            # If the cone does not intersect the volume for a given energy,
            # continue
            if check_valid_events and torch.all(~mask_cone):
                event.xsection[idx] = 0.0
                continue

            sca_compton_diff_xsection = torch.zeros_like(cos_delta_j)
            sca_compton_diff_xsection[mask_cone] = camera.get_compton_diff_xsection(
                self.energies[idx],
                # ENRIQUE: In Enrique thesis the cosbeta here is
                # the one of the known energy constant
                # for all voxels. Instead we use
                # cos_delta_j which is a matrix
                cos_delta_j[mask_cone],
            )

            # ENRIQUE: Different from Enrique's thesis: no V2V1^2 is considered here, the
            # constants, and the probability of escape of the secondary photon (i.e. no interaction)
            # The voxel's size is not taken into account either.
            kbl_j = torch.zeros_like(cos_delta_j)
            kbl_j[mask_cone] = (
                camera.sca_n_eff
                * sca_compton_diff_xsection[mask_cone]
                * self.m_e
                * 2
                * torch.pi
                # What Enrique did - use the E1 from the event
                / torch.pow(self.energies[idx] - event.Ee, 2)
            )

            int2Xsect = 0.0
            photo_x_section = camera.get_photo_diff_xsection(
                self.energies[idx] - event.Ee, DetectorType.ABS
            )
            photo_x_section_m_e = camera.get_photo_diff_xsection(
                self.m_e, DetectorType.ABS
            )

            pair_x_section = camera.get_pair_diff_xsection(
                self.energies[idx] - event.Ee, DetectorType.ABS
            )
            # Absorbition total is Photoelectric, partial absorbition is either
            # compton or pair production
            # Photoelectric
            if abs(self.energies[idx] - event.E0) < self.tol:
                int2Xsect = (
                    camera.abs_density * photo_x_section
                    # Include double absorption probability after pair creation
                    + camera.abs_density
                    * pair_x_section
                    * 2
                    * self.tol
                    * torch.pow(
                        (
                            1
                            - torch.exp(
                                -photo_x_section_m_e
                                * camera.abs_density
                                * (camera.sca_layers + camera.abs_layers)[
                                    event.layer_idx_V2
                                ].thickness
                                / 2.0
                            )
                        ),
                        2,
                    )
                )
            # Compton allowed
            else:
                cos_beta_2 = 1.0 - (
                    self.m_e
                    * event.Eg
                    / (
                        (self.energies[idx] - event.Ee)
                        * (self.energies[idx] - event.Ee - event.Eg)
                    )
                )
                if abs(cos_beta_2) <= 1.0:
                    abs_compton_diff_xsection = camera.get_compton_diff_xsection(
                        self.energies[idx],
                        cos_beta_2,
                    )

                    int2Xsect = (
                        camera.abs_n_eff
                        * abs_compton_diff_xsection
                        * self.m_e
                        * pi
                        * 4
                        # dE = tol * 2
                        # ENRIQUE: dE is not documented in the original C++ code
                        * self.tol
                        / (torch.pow(self.energies[idx] - event.E0, 2))
                    )

                # Test for pair production
                if (
                    abs(self.energies[idx] - (event.E0 + (2 * self.m_e))) < 2 * self.tol
                ):  # with double escape
                    int2Xsect += (
                        camera.abs_density
                        * pair_x_section
                        * torch.exp(
                            -photo_x_section_m_e
                            * camera.abs_density
                            * (camera.sca_layers + camera.abs_layers)[
                                event.layer_idx_V2
                            ].thickness
                        )
                    )

                elif (
                    abs(self.energies[idx] - (event.E0 + self.m_e)) < 2 * self.tol
                ):  # with single escape
                    int2Xsect += (
                        2
                        * camera.abs_density
                        * pair_x_section
                        * (
                            torch.exp(
                                -photo_x_section_m_e
                                * camera.abs_density
                                * (camera.sca_layers + camera.abs_layers)[
                                    event.layer_idx_V2
                                ].thickness
                                / 2.0
                            )
                            # ENRIQUE: why difference of two exponentials. Why not
                            # multiplication of probabilities?. and exp * (1-exp)?
                            - torch.exp(
                                --photo_x_section_m_e
                                * camera.abs_density
                                * (camera.sca_layers + camera.abs_layers)[
                                    event.layer_idx_V2
                                ].thickness
                            )
                        )
                    )
            if int2Xsect == 0.0:
                event.xsection[idx] = 0.0
                self.line.values[idx] = 0.0
                continue

            # Multiply by exponential terms
            # See List mode em reconstruction of Compton Sctter Camera Images in 3D.
            # Wilderman et al. for help and Enrique's thesis.
            # Probabilities for the first photon to reach the middle of the
            # scatterer layer
            kbl_j[mask_cone] *= torch.exp(
                # zd11
                -camera.get_total_diff_xsection(
                    self.energies[idx], event.detector_type_V1
                )
                # Attenuation coefficient is proportinal to the density
                # (which depends on the medium's physical state)
                * (
                    camera.sca_density
                    if event.detector_type_V1 == DetectorType.SCA
                    else camera.abs_density
                )
                # The thickness. layer_idx_V1 is in a defined layer i.e. it
                # will not be in a None absortber layer as this is checked
                # beforehand
                * (camera.sca_layers + camera.abs_layers)[event.layer_idx_V1].thickness
                # Assume the interaction is the middle of the material in a
                # orthogonal line?
                # TODO: take the incident angle into account?
                / 2
            )

            # zd12
            # Probability of of going out of the scatterer layer and to the
            # absorber
            kbl_j[mask_cone] *= torch.exp(
                -camera.get_total_diff_xsection(
                    self.energies[idx] - event.Ee, event.detector_type_V1
                )
                * (
                    camera.sca_density
                    if event.detector_type_V1 == DetectorType.SCA
                    else camera.abs_density
                )
                * (camera.sca_layers + camera.abs_layers)[event.layer_idx_V1].thickness
                / 2
            )
            # Attenuation in planes not triggered
            # Before scatterer
            # TODO: If first interaction can be in the absorber. In this case, we
            # would need to compute the number of scatteres the gamma went
            # through without interactions
            # Probability of first and 2nd photon to go through not triggered
            # layers, respectivelly
            if event.layer_idx_V1 > 0:
                kbl_j[mask_cone] *= torch.exp(
                    -(
                        camera.get_total_diff_xsection(
                            self.energies[idx], event.detector_type_V1
                        )
                        * (
                            camera.sca_density
                            if event.detector_type_V1 == DetectorType.SCA
                            else camera.abs_density
                        )
                        * event.layer_idx_V1
                        * (camera.sca_layers + camera.abs_layers)[
                            event.layer_idx_V1
                        ].thickness
                    )
                )
            # After scatterer
            # TODO: This works only for 1 single absorber below the scatterers
            # But what if the interaction is in an absorber on the sides. If
            # that happens, the scatterers below may needs to be ignored
            kbl_j[mask_cone] *= torch.exp(
                -(
                    camera.get_total_diff_xsection(
                        self.energies[idx] - event.Ee, event.detector_type_V1
                    )
                    * (
                        camera.sca_density
                        if event.detector_type_V1 == DetectorType.SCA
                        else camera.abs_density
                    )
                    * abs(event.layer_idx_V2 - event.layer_idx_V1 - 1)
                    * (camera.sca_layers + camera.abs_layers)[
                        event.layer_idx_V1
                    ].thickness
                )
            )

            if self.compute_theta_j:
                kbl_j = kbl_j * self.model(cos_theta_j, rho_j)
            elif not self.do_nothing:
                kbl_j = kbl_j * self.model(rho_j)

            # Gauss
            self.line.values[idx][mask_cone] = (
                int2Xsect
                * kbl_j[mask_cone]
                * torch.exp(
                    -(self.line.values[idx][mask_cone] ** 2) / (2 * self.sigma_beta**2)
                )
            )

        if check_valid_events and torch.all(event.xsection == 0.0):
            raise ValueError(
                f"The cone does not intersect the volume for event {event.id} for all tried energies"
            )

        return self.line

    def read_constants(self, constants_filename: str) -> dict:
        try:
            with open(constants_filename, "r") as fh:
                return yaml.safe_load(fh)
        except IOError as e:
            logger.fatal(f"Failed to open the constants file: {e}")
            sys.exit(1)

    @staticmethod
    def create_mesh_axes(
        x_range: tuple,
        x_steps: int,
        y_range: tuple,
        y_steps: int,
        z_range: tuple,
        z_steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Sample points along each volume dimension. use voxel size to center
        # the points on the voxels
        return (
            torch.linspace(*x_range, x_steps),
            torch.linspace(*y_range, y_steps),
            torch.linspace(*z_range, z_steps),
        )
