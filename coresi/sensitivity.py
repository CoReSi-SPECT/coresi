# SPDX-FileCopyrightText: 2024 Vincent Lequertier <vincent@vl8r.eu>, Voichita Maxim <voichita.maxim@creatis.insa-lyon.fr>
# CREATIS Laboratory, INSA Lyon, France
# SPDX-License-Identifier: MIT

import random
from logging import getLogger
from typing import Callable
import torch, torchvision

from coresi.camera import Camera, generate_random_angle
from coresi.event import Event
from coresi.image import Image
from coresi.point import Point
from coresi.utils import generate_random_point, generate_weighted_random_point

_ = torch.set_grad_enabled(False)

logger = getLogger("CORESI")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def block(
    cameras: list[Camera],
    volume_config: dict,
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
):
    """The sensitivity is the solid angle of scatterer bounding box"""
    cameras = [cameras[0]]
    # As computed here the sensitivity does not depend on the energy
    sensitivity_vol = Image(1, volume_config)
    # Put the volume in the coordinate system of the camera
    for camera in cameras:
        points = torch.tensordot(
            (
                torch.tensor(
                    [
                        Point(x[idx], y[idx2], z[idx3])
                        for idx in range(len(x))
                        for idx2 in range(len(y))
                        for idx3 in range(len(z))
                    ]
                )
                - camera.origin
            ),
            torch.tensor((camera.Ox, camera.Oy, camera.Oz), dtype=torch.double),
            dims=1,
        ).to(device)
        # 3rd dimension is z
        D = torch.abs(points[:, 2] - camera.sca_centre.z)
        sq = torch.sqrt(
            # 1st dimension is x
            torch.pow(points[:, 0] - camera.sca_centre.x, 2)
            # 2nd dimension is y
            + torch.pow(points[:, 1] - camera.sca_centre.y, 2)
            + torch.pow(D, 2)
        )
        sensitivity_vol.values += (
            D / sq**3 * camera.sca_layers[0].dim.x * camera.sca_layers[0].dim.y
        ).reshape(sensitivity_vol.values.shape)

    return sensitivity_vol.values


def attenuation_exp(
    cameras: list[Camera],
    volume_config: dict,
    energies: list[float],
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
):
    sensitivity_vol = Image(len(energies), volume_config)
    cameras = [cameras[0]]

    for camera in cameras:
        b, d = camera.sca_layers[0].dim.x / 2, camera.sca_layers[0].dim.y / 2
        a, c = -b, -d
        Nx = 100
        Ny = 100
        hx = (b - a) / Nx
        hy = (d - c) / Ny
        # Put the volume in the coordinate system of the camera
        points = torch.tensordot(
            (
                torch.tensor(
                    [
                        Point(x[idx1], y[idx2], z[idx3])
                        for idx1 in range(len(x))
                        for idx2 in range(len(y))
                        for idx3 in range(len(z))
                    ]
                )
                - camera.origin
            ),
            torch.tensor((camera.Ox, camera.Oy, camera.Oz), dtype=torch.double),
            dims=1,
        ).to(device)
        for idx_energy, energy in enumerate(energies):
            # Only compute sensitivity for sca layers, assume probabilities for
            # hits in absorber the same. Also assume first hit in sca. To be improved ...
            for idx_layer, layer in enumerate(camera.sca_layers):
                # for idx_layer, layer in enumerate(camera.sca_layers + camera.abs_layers):
                # Compton linear attenuation coefficient (incoherent scattering)
                mu_Compton = (
                    camera.get_incoherent_diff_xsection(energy, layer.detector_type)
                    * camera.sca_density
                )
                # total linear attenuation coefficient
                mu_total = (
                    camera.get_total_diff_xsection(energy, layer.detector_type)
                    * camera.sca_density
                )
                # 3rd dimension is z
                D = torch.abs(points[:, 2] - layer.center.z)
                rect = 0.0
                for m in range(0, Nx):
                    for n in range(0, Ny):
                        sq = (
                            (
                                (a + (m + 0.5) * hx - points[:, 0])
                                * (a + (m + 0.5) * hx - points[:, 0])
                            )
                            + (
                                (c + (n + 0.5) * hy - points[:, 1])
                                * (c + (n + 0.5) * hy - points[:, 1])
                            )
                            + D**2
                        )
                        # rect
                        # += 1.0/sq*exp(-mu*L*(layer-1)*sqrt(sq)/D);
                        rect += (
                            D
                            * torch.pow(sq, -1.5)
                            * torch.exp(
                                -mu_total
                                * camera.sca_layers[0].dim.z
                                * idx_layer
                                * torch.sqrt(sq)
                                / D
                            )
                            * (
                                1
                                - torch.exp(
                                    -mu_Compton
                                    * camera.sca_layers[0].dim.z
                                    * torch.sqrt(sq)
                                    / D
                                )
                            )
                        )
                sensitivity_vol.values[idx_energy] += (
                    rect.reshape(sensitivity_vol.values[idx_energy].shape) * hx * hy
                )
    return sensitivity_vol.values


def lyon_4D(
    cameras: list[Camera],
    volume_config: dict,
    energies: list[float],
    SM_line: Callable[[Event, bool], Image],
    mc_samples: int = 1,
):
    """Compute a system matrix by computing the probability of a random gammas
    reaching the cameras detectos"""
    sensitivity_vol = Image(len(energies), volume_config)
    angles = torch.arange(0, 181, 1).deg2rad()
    cdf_compton = cameras[0].cdf_compton_diff_xsection_dtheta(energies, angles)
    for camera in cameras:
        for idx_energy, energy in enumerate(energies):
            valid_events = 0
            while valid_events < mc_samples:
                sca = random.choice(camera.sca_layers)
                absorber = random.choice(camera.abs_layers)
                x1 = generate_random_point(sca.dim, sca.center, 1)[0]
                x2 = generate_random_point(absorber.dim, absorber.center, 1)[0]
                beta = generate_random_angle(cdf_compton, angles, idx_energy)
                E_gamma = energy / (1 + (energy / 511.0) * (1 - torch.cos(beta)))
                # The Event class does the conversion to centimeters
                x1 = x1 * 10
                x2 = x2 * 10
                # Forge an event with the GATE line's format
                line = f"2\t1\t{x1[0]}\t{x1[1]}\t{x1[2]}\t{energy - E_gamma}\t2\t{x2[0]}\t{x2[1]}\t{x2[2]}\t{E_gamma}\t3\t0\t0\t0\t0"
                try:
                    event = Event(
                        0,
                        line,
                        energies,
                        Point(*volume_config["volume_centre"]),
                        Point(*volume_config["volume_dimensions"]),
                    )
                    event.set_camera_index([camera])
                except ValueError as e:
                    logger.debug(f"Skipping event {line.strip()} REASON: {e}")
                    continue
                try:
                    result = SM_line(event, True).values
                    # Because we want a sensitivity for a given energy, ensure
                    # that the SM line for the given energy is non-zero
                    if result[idx_energy].any():
                        sensitivity_vol.values[idx_energy] = (
                            sensitivity_vol.values[idx_energy] + result[idx_energy]
                        )
                    else:
                        raise ValueError(f"Got zeros for energy {energy} keV")
                except ValueError as e:
                    logger.debug(f"Skipping forged event {line.strip()} REASON: {e}")
                    continue
                valid_events += 1

    # We only do the sum here rather than the average because it's unlikely we
    # get high values as the chance the cone goes perfectly through the
    # voxel is low
    return sensitivity_vol.values


def sm_like(
    cameras: list[Camera],
    volume_config: dict,
    energies: list[float],
    SM_line: Callable[[Event, bool], Image],
    mc_samples: int = 1,
):
    """Compute a system matrix by computing the probability of a random gammas
    reaching the cameras detectos"""
    sensitivity_vol = Image(len(energies), volume_config)
    angles = torch.arange(0, 181, 1).deg2rad()
    # cdf_compton = cameras[0].cdf_compton_diff_xsection(energies, angles)
    # volume = Image(len(energies), volume_config, init="ones")
    for camera in cameras:
        for idx_energy, energy in enumerate(energies):
            valid_events = 0
            while valid_events < mc_samples:
                # x0, k = generate_weighted_random_point(volume, 1)
                x0 = generate_random_point(
                    sensitivity_vol.dim_in_cm, sensitivity_vol.center, 1
                )[0]
                sca = random.choice(camera.sca_layers)
                absorber = random.choice(camera.abs_layers)
                x1 = generate_random_point(sca.dim, sca.center, 1)[0]
                x1 = (  # Convert the random point into camera coordinates
                    x1[0] * camera.Ox
                    + x1[1] * camera.Oy
                    + x1[2] * camera.Oz
                    + camera.origin
                )
                x2 = generate_random_point(absorber.dim, absorber.center, 1)[0]
                x2 = (  # Convert the random point into camera coordinates
                    x2[0] * camera.Ox
                    + x2[1] * camera.Oy
                    + x2[2] * camera.Oz
                    + camera.origin
                )
                r1 = x1 - x0
                r2 = x2 - x1
                cosbeta = (r1 * r2).sum(axis=0) / (
                    torch.linalg.norm(r1, 2, axis=0) * torch.linalg.norm(r2, 2, axis=0)
                )
                E_gamma = energy / (1 + (energy / 511.0) * (1 - cosbeta))
                # The Event class does the conversion to centimeters
                x1 = x1 * 10
                x2 = x2 * 10
                # Forge an event with the GATE line's format
                line = f"2\t1\t{x1[0]}\t{x1[1]}\t{x1[2]}\t{energy - E_gamma}\t2\t{x2[0]}\t{x2[1]}\t{x2[2]}\t{E_gamma}\t3\t0\t0\t0\t0"
                try:
                    event = Event(
                        0,
                        line,
                        energies,
                        Point(*volume_config["volume_centre"]),
                        Point(*volume_config["volume_dimensions"]),
                    )
                    event.set_camera_index([camera])
                except ValueError as e:
                    logger.debug(f"Skipping event {line.strip()} REASON: {e}")
                    continue
                try:
                    result = SM_line(event, True).values
                    # Because we want a sensitivity for a given energy, ensure
                    # that the SM line for the given energy is non-zero
                    if result[idx_energy].any():
                        sensitivity_vol.values[idx_energy] = (
                            sensitivity_vol.values[idx_energy] + result[idx_energy]
                        )
                    else:
                        raise ValueError(f"Got zeros for energy {energy} keV")
                except ValueError as e:
                    logger.debug(f"Skipping forged event {line.strip()} REASON: {e}")
                    continue
                valid_events += 1

    # We only do the sum here rather than the average because it's unlikely we
    # get high values as the chance the cone goes perfectly through the
    # voxel is low

    # sm_like Monte Carlo approach creates irregularities in the final sensitivity
    # matrix,therefore we perform a smoothing with a gaussian kernel
    gaussian_blur = torchvision.transforms.GaussianBlur(kernel_size=9, sigma=2.0)
    sens_vol = sensitivity_vol.values.permute(3, 0, 1, 2)
    sens_vol_smooth = gaussian_blur(sens_vol).permute(1, 2, 3, 0)

    sensitivity_vol.values = sens_vol_smooth

    return sensitivity_vol.values
