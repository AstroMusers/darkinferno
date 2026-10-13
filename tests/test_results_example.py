import importlib.util
from pathlib import Path


EXAMPLE_PATH = Path(__file__).resolve().parents[1] / "examples" / "reproduce_cappiello_daylan_2025.py"
SPECIFICATION = importlib.util.spec_from_file_location(
    "darkinferno_full_results_example", EXAMPLE_PATH
)
example = importlib.util.module_from_spec(SPECIFICATION)
SPECIFICATION.loader.exec_module(example)


def test_full_results_example_runs_all_analysis_scripts(tmp_path, monkeypatch):
    calls = []

    def record_run(command, **options):
        calls.append((command, options))

    monkeypatch.setattr(example.subprocess, "run", record_run)
    output_dir = tmp_path / "paper_results"

    returned_path = example.run_results(output_dir, typefileplot="pdf", typeverb=0)

    assert returned_path == output_dir
    assert tuple(Path(command[1]).name for command, _ in calls) == example.RESULT_SCRIPTS
    assert all(options["check"] for _, options in calls)
    assert all(options["cwd"] == example.REPOSITORY_ROOT for _, options in calls)
    assert all(command[command.index("--output-dir") + 1] == str(output_dir) for command, _ in calls)
    assert all(command[command.index("--typefileplot") + 1] == "pdf" for command, _ in calls)
    assert all(command[command.index("--typeverb") + 1] == "0" for command, _ in calls)