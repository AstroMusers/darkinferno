#!/usr/bin/env python3
"""Generate the current DarkInferno thermal and capture result suite using Gordion."""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

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
DEFAULT_OUTPUT_DIRECTORY = REPOSITORY_ROOT / "examples" / "visuals"


def source_hashes():
    """Record the actual thermal, capture, and plotting Python sources."""
    from darkinferno import physics
    import gordion

    sources = {}
    for label, directory in (("darkinferno", Path(physics.__file__).parent),
                              ("gordion", Path(gordion.__file__).parent),
                              ("examples", REPOSITORY_ROOT / "examples")):
        for path in sorted(directory.rglob("*.py")):
            name = f"{label}/{path.relative_to(directory)}"
            print(f"Reading from {name}...")
            sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return sources


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="directory for all generated figures and tables",
    )
    parser.add_argument("--typefileplot", choices=("png", "pdf"), default="png")
    parser.add_argument("--typeplotback", choices=("white", "dark"), default="white")
    parser.add_argument("--typeverb", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--time-tag", help="output timestamp YYYYMMDD_HHMMSS with optional microseconds")
    return parser.parse_args(argv)


def run_results(output_dir, typefileplot="png", typeverb=1, typeplotback="white", *, time_tag=None):
    """Run every DarkInferno result-producing example with shared output settings."""

    if typefileplot not in {"png", "pdf"} or typeplotback not in {"white", "dark"} or typeverb not in {0, 1, 2}:
        raise ValueError("Invalid figure format, theme or verbosity")
    if time_tag is None:
        time_tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    if not isinstance(time_tag, str) or not re.fullmatch(r"\d{8}_\d{6}(?:_\d{6})?", time_tag):
        raise ValueError("time_tag must be YYYYMMDD_HHMMSS with optional microseconds")
    output_dir = Path(output_dir)
    manifest_path = output_dir / f"cappiello_daylan_2025_{time_tag}.json"
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite {manifest_path}")
    sources = source_hashes()
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="darkinferno-results-") as temporary:
        staging = Path(temporary)
        for script_name in RESULT_SCRIPTS:
            script_path = REPOSITORY_ROOT / "examples" / script_name
            before = set(staging.iterdir())
            command = [sys.executable, str(script_path), "--output-dir", str(staging),
                       "--typefileplot", typefileplot, "--typeplotback", typeplotback,
                       "--typeverb", str(typeverb)]
            print(f"Running {script_name}...")
            subprocess.run(command, check=True, cwd=REPOSITORY_ROOT)
            if not set(staging.iterdir()) - before:
                raise RuntimeError(f"{script_name} produced no new result files")
        if source_hashes() != sources:
            raise RuntimeError("Scientific source changed during the run; public outputs were left unchanged")
        products = {path: output_dir / f"{path.stem}_{time_tag}{path.suffix}"
                    for path in sorted(staging.iterdir()) if path.is_file()}
        for destination in products.values():
            if destination.exists():
                raise FileExistsError(f"Refusing to overwrite {destination}")
        manifest = {"time_tag": time_tag, "typefileplot": typefileplot,
                    "typeplotback": typeplotback, "scripts": list(RESULT_SCRIPTS),
                    "source_sha256": sources, "python_version": sys.version,
                    "dependency_versions": {name: version(name) for name in ("numpy", "scipy", "astropy")},
                    "outputs": {destination.name: hashlib.sha256(source.read_bytes()).hexdigest()
                                for source, destination in products.items()}}
        for source, destination in products.items():
            print(f"Writing to {destination}...")
            with destination.open("xb") as output:
                output.write(source.read_bytes())
        print(f"Writing to {manifest_path}...")
        with manifest_path.open("x", encoding="utf-8") as output:
            json.dump(manifest, output, indent=2)
            output.write("\n")
    return output_dir


def main(argv=None):
    arguments = parse_arguments(argv)
    run_results(arguments.output_dir, arguments.typefileplot, arguments.typeverb, arguments.typeplotback,
                time_tag=arguments.time_tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())