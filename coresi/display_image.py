import argparse
from pathlib import Path

import git
import numpy as np
import torch
import yaml

from coresi.image import Image

parser = argparse.ArgumentParser(description="CORESI")
repo = git.Repo(search_parent_directories=True)
commit = repo.git.rev_parse("HEAD", short=True)

parser.add_argument("-i", "--image", type=Path, required=True)
parser.add_argument(
    "-c",
    "--config",
    default="config.yaml",
    help="Path to the configuration file",
    type=Path,
)

parser.add_argument(
    "-s",
    "--slice",
    default=0,
    help="Slice number",
    type=int,
)
parser.add_argument(
    "-p",
    "--projection",
    default="z",
    help="Projection x, y or z",
    type=str,
)
parser.add_argument(
    "--cpp",
    action=argparse.BooleanOptionalAction,
    help="Use this if the file comes from the C++ version of CORESI",
    type=bool,
)
args = parser.parse_args()


def display():
    with open(args.config, "r") as fh:
        config = yaml.safe_load(fh)

    image = Image(len(config["E0"]), config["volume"])

    if args.cpp:
        image.values = torch.from_numpy(
            np.fromfile(args.image)
            .reshape(image.values.shape)
            .transpose(-4, -2, -3, -1)
        )
    else:
        image.values = torch.load(args.image, map_location=torch.device("cpu"))

    for e in range(image.values.shape[0]):
        if args.projection == "z":
            image.display_z(
                energy=e,
                title=f" {str(config['E0'][e])} keV" + (" CPP" if args.cpp else ""),
                slice=args.slice,
            )
        if args.projection == "x":
            image.display_x(
                energy=e,
                title=f" {str(config['E0'][e])} keV" + (" CPP" if args.cpp else ""),
                slice=args.slice,
            )
        if args.projection == "y":
            image.display_y(
                energy=e,
                title=f" {str(config['E0'][e])} keV" + (" CPP" if args.cpp else ""),
                slice=args.slice,
            )


if __name__ == "__main__":
    display()
