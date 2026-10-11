import argparse

from Examples.example import Example, list_examples


def build_parser():
    parser = argparse.ArgumentParser(
        description="Power Flow Case Comparison Engine - Examples"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True
    )

    # ------------------------------------------------------------------
    # CREATE / RUN EXAMPLE
    # ------------------------------------------------------------------
    create_parser = subparsers.add_parser(
        "run",
        help="Create and run a predefined power-flow example"
    )

    create_parser.add_argument(
        "example_name",
        choices=list_examples(),
        help="Name of the example to run"
    )

    create_parser.add_argument(
        "--create-topology-mismatch",
        action="store_true",
        help="Modify Case B to create a topology mismatch"
    )

    create_parser.add_argument(
        "--increase-load-by",
        type=float,
        default=1.0,
        help="Multiply Case B load by this factor (default: 1.0)"
    )

    create_parser.set_defaults(func=run_example)

    # ------------------------------------------------------------------
    # LIST EXAMPLES
    # ------------------------------------------------------------------
    list_parser = subparsers.add_parser(
        "list",
        help="List all available examples"
    )

    list_parser.set_defaults(func=list_available_examples)

    return parser


def run_example(args):
    options = {
        "create_topology_mismatch": args.create_topology_mismatch,
        "increase_load_by": args.increase_load_by,
    }

    Example.create(
        example_name=args.example_name,
        **options
    )


def list_available_examples(args):
    print("Available examples:")

    for example_name in list_examples():
        print(f"  - {example_name}")


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
