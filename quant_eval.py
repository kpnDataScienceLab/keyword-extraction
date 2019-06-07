import argparse
import json


def main(ds_name):
    if ds_name == '500N-KPCrowd':
        with open()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ds",
        type=str,
        default='500N-KPCrowd',
        choices=['500N-KPCrowd'],
        help="Name of the dataset to use"
    )

    flags = parser.parse_args()
    main(flags.ds_name)
