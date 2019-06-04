import pickle as pkl
import argparse


def view(flags):
    with open('processed/' + flags.file_name + '.pkl', 'rb') as handle:
        keys = pkl.load(handle)

    if flags.text:
        print()
        print(keys[flags.id]['text'])

    print()
    for idx, k in enumerate(keys[flags.id]['keywords'][0:flags.max_keywords]):
        print(f"{idx + 1}) {k}")

    print()
    print(f"{len(keys[flags.id]['keywords'])} total keywords")
    print()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--file_name",
        type=str,
        help="Name of the file containing the extracted keywords",
    )
    parser.add_argument(
        "--id",
        type=int,
        default=0,
        help="ID of document to examine"
    )
    parser.add_argument(
        "--max_keywords",
        type=int,
        default=100,
        help="Maximum amount of keywords to be printed"
    )
    parser.add_argument(
        "--text",
        help="Whether to show the text or not",
        action="store_true"
    )

    flags = parser.parse_args()
    view(flags)


main()
