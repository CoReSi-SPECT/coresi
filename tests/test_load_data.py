import os
import sys
import unittest
from pathlib import Path

import yaml

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")

from camera import setup_cameras
from data import read_data_file
from event import Event

test_dir = Path(os.path.dirname(os.path.realpath(__file__)))

with open(test_dir / "test_config.yaml", "r") as fh:
    config = yaml.safe_load(fh)

cameras = setup_cameras(config["cameras"])


class LoadData(unittest.TestCase):
    def test_read_data_file(self):
        events = read_data_file(
            config["data_file"],
            n_events=-1,
            E0=config["E0"],
            cameras=cameras,
            energy_range=config["energy_range"],
            remove_out_of_range_energies=config["remove_out_of_range_energies"],
            start_position=0,
        )

        self.assertEqual(len(events), 19744, "Wrong number of events")

        events = read_data_file(
            config["data_file"],
            n_events=2,
            E0=config["E0"],
            cameras=cameras,
            energy_range=config["energy_range"],
            remove_out_of_range_energies=config["remove_out_of_range_energies"],
            start_position=24,
        )

        self.assertEqual(len(events), 2, "Wrong number of events")

        events = read_data_file(
            config["data_file"],
            n_events=1,
            E0=config["E0"],
            cameras=cameras,
            energy_range=config["energy_range"],
            remove_out_of_range_energies=config["remove_out_of_range_energies"],
            start_position=0,
        )

        self.assertEqual(len(events), 1, "Wrong number of events")

        with self.assertRaises(ValueError):
            with open(
                config["data_file"],
                "r",
            ) as data_fh:
                for line_n, line in enumerate(data_fh):
                    if line_n == 13:
                        Event(line_n, line, -1)

    def test_dat_data(self):
        events = read_data_file(
            config["data_file"],
            n_events=26,
            E0=config["E0"],
            cameras=cameras,
            energy_range=config["energy_range"],
            remove_out_of_range_energies=config["remove_out_of_range_energies"],
            start_position=0,
        )
        self.assertEqual(
            [event.E0 for event in events],
            [
                140.00008,
                140.000485,
                139.99968,
                139.9996,
                139.9999,
                140.0004,
                140.00038,
                140.00014000000002,
                139.99993,
                140.00025,
                140.00041,
                140.0003,
                140.0004,
                140.000337,
                140.000241,
                140.00030999999998,
                140.0,
                140.0003,
                140.00043399999998,
                139.99957,
                139.9997,
                140.00027,
                139.99982,
                139.9995,
            ],
            "Wrong E0",
        )

        self.assertEqual(
            [event.beta for event in events],
            [
                0.4182565668709084,
                0.2238488843404098,
                0.7147145964617408,
                0.5999558863741418,
                0.8808123012342127,
                0.47868440364425235,
                0.5784756952332849,
                0.631358266335426,
                0.7377916241046396,
                0.6207770435285515,
                0.3151071385648606,
                1.1298650244328245,
                1.3574997629002692,
                0.13228923578335036,
                0.15951071629591188,
                0.3404428331102263,
                0.9349821353202604,
                0.896995259466335,
                0.2119934030811821,
                0.5545581944650999,
                0.43385891972508167,
                0.23882368061438372,
                0.4166230049448475,
                0.9874767882413626,
            ],
            "Wrong beta",
        )


if __name__ == "__main__":
    unittest.main()
