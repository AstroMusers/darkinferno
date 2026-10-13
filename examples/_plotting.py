import argparse
from pathlib import Path
import sys

import matplotlib


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tdpy import narrate


BACKGROUND = "#ffffff"
INK = "#172126"
EARTH_COLORS = ("#0b5d6b", "#288c8c", "#e9b44c", "#d1495b", "#713e5a")
MOON_COLORS = ("#343a40", "#737b82", "#d6a13d", "#a44a3f", "#6f4e7c")
MARS_COLORS = ("#a63d40", "#d97941", "#5c4742", "#e2a66f", "#7c2d33")
PAPER_COLUMN_WIDTH_IN = 3.5  # [inch]
PAPER_SINGLE_PLOT_HEIGHT_IN = 4.0  # [inch]
MAIN_TEXT_SIZE_PT = 10.5  # [point]


def parse_plot_args(description):
    """Parse shared plot-output and narration arguments."""

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--output-dir",
        default=REPOSITORY_ROOT / "figures",
        help="directory for the generated figure",
    )
    parser.add_argument(
        "--typefileplot",
        choices=("png", "pdf"),
        default="png",
        help="figure file type (default: %(default)s)",
    )
    parser.add_argument(
        "--typeverb",
        type=int,
        choices=(0, 1, 2),
        default=1,
        help="narration level: 0 critical, 1 nominal, 2 verbose (default: %(default)s)",
    )
    arguments = parser.parse_args()

    return arguments


def configure_plotting():
    """Configure publication-scale plotting on a white background."""

    plt.rcParams.update(
        {
            "figure.facecolor": BACKGROUND,
            "axes.facecolor": BACKGROUND,
            "savefig.facecolor": BACKGROUND,
            "text.usetex": False,
            "font.family": "DejaVu Sans",
            "font.size": MAIN_TEXT_SIZE_PT,
            "axes.labelsize": MAIN_TEXT_SIZE_PT,
            "xtick.labelsize": MAIN_TEXT_SIZE_PT,
            "ytick.labelsize": MAIN_TEXT_SIZE_PT,
            "legend.fontsize": MAIN_TEXT_SIZE_PT,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.edgecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "legend.frameon": False,
            "legend.labelcolor": INK,
            "lines.linewidth": 2.3,
            "figure.dpi": 120,
            "savefig.dpi": 300,
        }
    )


def single_plot(height_in=PAPER_SINGLE_PLOT_HEIGHT_IN):
    """Create one paper-width plot."""

    figure_axis = plt.subplots(
        figsize=(PAPER_COLUMN_WIDTH_IN, height_in), layout="constrained"
    )

    return figure_axis


def finish_plot(
    figure,
    axis,
    output_directory,
    stem,
    typefileplot,
    legend_location,
    legend_columns=1,
    legend_anchor=None,
    **axis_settings,
):
    """Apply labels and legend, then write one figure."""

    axis.set(**axis_settings)
    axis.legend(loc=legend_location, ncols=legend_columns, bbox_to_anchor=legend_anchor)
    save_figure(figure, output_directory, stem, typefileplot=typefileplot)


def figure_path(output_directory, stem, typefileplot="png"):
    """Construct a validated figure output path."""

    if typefileplot not in {"png", "pdf"}:
        raise ValueError("typefileplot must be 'png' or 'pdf'")
    output_path = Path(output_directory) / f"{stem}.{typefileplot}"

    return output_path


def figure_already_exists(output_directory, stem, typefileplot="png"):
    """Report whether a requested figure already exists."""

    output_path = figure_path(output_directory, stem, typefileplot)
    if output_path.exists():
        print(f"{output_path} already exists...")
        exists = True
    else:
        exists = False

    return exists


def save_figure(figure, output_directory, stem, typefileplot="png"):
    """Write a PNG at 300 dpi or a vector PDF."""

    from tdpy.plotting import save_figure as tdpy_save_figure

    output_path = figure_path(output_directory, stem, typefileplot)
    tdpy_save_figure(figure, output_path, typefileplot=typefileplot, close_figure=True)
