import importlib.util
import hashlib
import json
from pathlib import Path

import pytest


EXAMPLE_PATH = Path(__file__).resolve().parents[1] / "examples" / "reproduce_cappiello_daylan_2025.py"
SPECIFICATION = importlib.util.spec_from_file_location(
    "darkinferno_full_results_example", EXAMPLE_PATH
)
example = importlib.util.module_from_spec(SPECIFICATION)
SPECIFICATION.loader.exec_module(example)


@pytest.mark.parametrize('typeplotback', ['white', 'dark'])
def test_full_results_example_runs_all_analysis_scripts(tmp_path, monkeypatch, typeplotback):
    calls = []

    def record_run(command, **options):
        calls.append((command, options))
        staging = Path(command[command.index("--output-dir") + 1])
        (staging / f"{Path(command[1]).stem}.pdf").write_bytes(b"test output")

    monkeypatch.setattr(example.subprocess, "run", record_run)
    output_dir = tmp_path / "paper_results"

    returned_path = example.run_results(output_dir, typefileplot="pdf", typeverb=0, typeplotback=typeplotback)

    assert returned_path == output_dir
    assert tuple(Path(command[1]).name for command, _ in calls) == example.RESULT_SCRIPTS
    assert all(options["check"] for _, options in calls)
    assert all(options["cwd"] == example.REPOSITORY_ROOT for _, options in calls)
    assert all(command[command.index("--output-dir") + 1] != str(output_dir) for command, _ in calls)
    assert all(command[command.index("--typefileplot") + 1] == "pdf" for command, _ in calls)
    assert all(command[command.index("--typeverb") + 1] == "0" for command, _ in calls)
    assert all(command[command.index("--typeplotback") + 1] == typeplotback for command, _ in calls)
    manifest = json.loads(next(output_dir.glob("*.json")).read_text())
    assert len(manifest["outputs"]) == len(example.RESULT_SCRIPTS)
    assert any(name.startswith("gordion/") for name in manifest["source_sha256"])
    for filename, digest in manifest["outputs"].items():
        assert hashlib.sha256((output_dir / filename).read_bytes()).hexdigest() == digest


def test_results_failure_preserves_public_outputs(tmp_path, monkeypatch):
    old = tmp_path / "earlier.png"
    old.write_bytes(b"earlier result")
    calls = []

    def fail_during_results(command, **options):
        calls.append(command)
        if len(calls) == 2:
            raise example.subprocess.CalledProcessError(1, command)
        staging = Path(command[command.index("--output-dir") + 1])
        (staging / "partial.png").write_bytes(b"new partial result")

    monkeypatch.setattr(example.subprocess, "run", fail_during_results)
    with pytest.raises(example.subprocess.CalledProcessError):
        example.run_results(tmp_path)
    assert list(tmp_path.iterdir()) == [old]
    assert old.read_bytes() == b"earlier result"


def test_results_timestamp_collision_rejected_before_running(tmp_path, monkeypatch):
    manifest = tmp_path / "cappiello_daylan_2025_20261025_220000.json"
    manifest.write_text("earlier manifest")
    monkeypatch.setattr(example.subprocess, "run", lambda *args, **kwargs: pytest.fail("Existing run"))
    with pytest.raises(FileExistsError):
        example.run_results(tmp_path, time_tag="20261025_220000")
    assert manifest.read_text() == "earlier manifest"


def test_results_source_change_prevents_publication(tmp_path, monkeypatch):
    hashes = iter([{"thermal.py": "before"}, {"thermal.py": "after"}])
    monkeypatch.setattr(example, "source_hashes", lambda: next(hashes))

    def write_result(command, **options):
        staging = Path(command[command.index("--output-dir") + 1])
        (staging / f"{Path(command[1]).stem}.png").write_bytes(b"staged output")

    monkeypatch.setattr(example.subprocess, "run", write_result)
    with pytest.raises(RuntimeError, match="source changed"):
        example.run_results(tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('typeplotback', ['white', 'dark'])
def test_example_plot_helpers_render_requested_theme(tmp_path, typeplotback):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import to_rgba
    from PIL import Image
    from tdpy.plotting import plot_background_colors, plot_style

    specification = importlib.util.spec_from_file_location(
        'darkinferno_plot_helpers', EXAMPLE_PATH.with_name('_plotting.py')
    )
    plotting = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(plotting)
    background, foreground = plot_background_colors(typeplotback)
    with plot_style('white'):
        plotting.configure_plotting(typeplotback)
        figure, axis = plotting.single_plot()
        FigureCanvasAgg(figure)
        axis.plot([0, 1], [1, 0], color='red', label='data')
        plotting.finish_plot(figure, axis, tmp_path, 'theme', 'png', 'best', xlabel='x', ylabel='y')

        assert axis.get_facecolor() == to_rgba(background)
        assert axis.xaxis.label.get_color() == foreground
        assert plotting.foreground_color() == foreground
        assert all(text.get_color() == foreground for text in axis.get_legend().get_texts())
        assert axis.get_legend().get_frame_on()
        assert axis.get_legend().get_frame().get_alpha() == 1.0
        filename = 'theme_dark.png' if typeplotback == 'dark' else 'theme.png'
        with Image.open(tmp_path / filename) as image:
            assert image.convert('RGB').getpixel((0, 0)) == tuple(int(channel * 255) for channel in to_rgba(background)[:3])
        if typeplotback == 'dark':
            plotting.configure_plotting('white')
            assert plotting.figure_path(tmp_path, 'theme').name == 'theme.png'
            assert not plotting.figure_already_exists(tmp_path, 'theme')


def test_earth_evolution_uses_its_measured_temperature_range(tmp_path, monkeypatch):
    import runpy
    from types import SimpleNamespace
    import numpy as np

    monkeypatch.syspath_prepend(str(EXAMPLE_PATH.parent))
    namespace = runpy.run_path(str(EXAMPLE_PATH.with_name("plot_earth.py")))
    globals_used = namespace["main"].__globals__
    monkeypatch.setitem(globals_used, "parse_plot_args", lambda _: SimpleNamespace(
        output_dir=tmp_path, typefileplot="png", typeverb=0, typeplotback="white"))
    save = globals_used["finish_plot"]
    captured = []

    def check_limits(figure, axis, directory, stem, *args, **kwargs):
        save(figure, axis, directory, stem, *args, **kwargs)
        if stem == "02_earth_temperature_evolution":
            values = np.concatenate([np.asarray(line.get_ydata()) for line in axis.lines])
            lower, upper = axis.get_ylim()
            assert lower <= values.min() and upper >= values.max()
            assert upper - lower < 2 * np.ptp(values)
            captured.append(stem)

    monkeypatch.setitem(globals_used, "finish_plot", check_limits)
    namespace["main"]()
    assert captured == ["02_earth_temperature_evolution"]