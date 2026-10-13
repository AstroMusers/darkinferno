#!/usr/bin/env python3
"""Generate DarkInferno's full Cappiello & Daylan (2025) result suite."""

import argparse
from pathlib import Path
import subprocess
import sys

from tdpy.verbosity import print


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RESULT_SCRIPTS = (
    "plot_earth.py",
    "plot_moon.py",
    "plot_comparison.py",
    "plot_intermediates.py",
    "plot_convergence.py",
    "plot_particle_constraint.py",
)
DEFAULT_OUTPUT_DIRECTORY = REPOSITORY_ROOT / "visuals" / "cappiello_daylan_2025"


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="directory for all generated figures and tables",
    )
    parser.add_argument("--typefileplot", choices=("png", "pdf"), default="png")
    parser.add_argument("--typeverb", type=int, choices=(0, 1, 2), default=1)
    return parser.parse_args(argv)


def run_results(output_dir, typefileplot="png", typeverb=1):
    """Run every DarkInferno result-producing example with shared output settings."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for script_name in RESULT_SCRIPTS:
        script_path = REPOSITORY_ROOT / "examples" / script_name
        command = [
            sys.executable,
            str(script_path),
            "--output-dir",
            str(output_dir),
            "--typefileplot",
            typefileplot,
            "--typeverb",
            str(typeverb),
        ]
        print(f"Running {script_path}...")
        subprocess.run(command, check=True, cwd=REPOSITORY_ROOT)
    return output_dir


def main(argv=None):
    arguments = parse_arguments(argv)
    run_results(arguments.output_dir, arguments.typefileplot, arguments.typeverb)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())