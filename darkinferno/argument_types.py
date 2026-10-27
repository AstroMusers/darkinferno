"""Shared command-line argument validators for DarkInferno scripts."""

import argparse
import math
from pathlib import Path


def add_shared_arguments(parser, scenarios, output_help):
    """Add DarkInferno options shared by its command-line tools."""
    parser.add_argument(
        "--scenario",
        choices=sorted(scenarios),
        default="earth-inner-core",
        help="body and thermal scenario (default: %(default)s)",
    )
    parser.add_argument("--output", type=Path, help=output_help)
    parser.add_argument(
        "--typeverb",
        type=int,
        choices=(0, 1, 2),
        default=1,
        help="narration level: 0 critical, 1 nominal, 2 verbose (default: %(default)s)",
    )
    return parser


def finite_float(value):
    """Parse a finite floating-point argument."""

    number = float(value)
    if not math.isfinite(number):
        raise argparse.ArgumentTypeError("must be finite")
    return number


def positive_float(value):
    """Parse a positive floating-point argument."""

    number = finite_float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def positive_integer(value):
    """Parse a positive integer argument."""

    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def fraction(value):
    """Parse a fraction in the interval (0, 1]."""

    number = finite_float(value)
    if not 0 < number <= 1:
        raise argparse.ArgumentTypeError("must be greater than zero and at most one")
    return number