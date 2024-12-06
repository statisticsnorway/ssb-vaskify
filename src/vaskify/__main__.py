"""Command-line interface."""

import click


@click.command()
@click.version_option()
def main() -> None:
    """Vaskify."""


if __name__ == "__main__":
    main(prog_name="ssb-vaskify")  # pragma: no cover
