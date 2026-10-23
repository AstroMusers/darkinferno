import importlib.util
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

    monkeypatch.setattr(example.subprocess, "run", record_run)
    output_dir = tmp_path / "paper_results"

    returned_path = example.run_results(output_dir, typefileplot="pdf", typeverb=0, typeplotback=typeplotback)

    assert returned_path == output_dir
    assert tuple(Path(command[1]).name for command, _ in calls) == example.RESULT_SCRIPTS
    assert all(options["check"] for _, options in calls)
    assert all(options["cwd"] == example.REPOSITORY_ROOT for _, options in calls)
    assert all(command[command.index("--output-dir") + 1] == str(output_dir) for command, _ in calls)
    assert all(command[command.index("--typefileplot") + 1] == "pdf" for command, _ in calls)
    assert all(command[command.index("--typeverb") + 1] == "0" for command, _ in calls)
    assert all(command[command.index("--typeplotback") + 1] == typeplotback for command, _ in calls)


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
        filename = 'theme_dark.png' if typeplotback == 'dark' else 'theme.png'
        with Image.open(tmp_path / filename) as image:
            assert image.convert('RGB').getpixel((0, 0)) == tuple(int(channel * 255) for channel in to_rgba(background)[:3])
        if typeplotback == 'dark':
            plotting.configure_plotting('white')
            assert plotting.figure_path(tmp_path, 'theme').name == 'theme.png'
            assert not plotting.figure_already_exists(tmp_path, 'theme')