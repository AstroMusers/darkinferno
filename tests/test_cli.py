import subprocess
import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIRECTORY = REPOSITORY_ROOT / "darkinferno"


class CommandLineTests(unittest.TestCase):
    def run_script(self, script_name, *arguments):
        return subprocess.run(
            [sys.executable, str(SCRIPTS_DIRECTORY / script_name), *arguments],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_radius_help_does_not_require_scientific_dependencies(self):
        result = self.run_script("darkinfernoradius.py", "--help")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("log10_dark_matter_mass", result.stdout)
        self.assertIn("log10_cross_section", result.stdout)

    def test_time_help_describes_physical_inputs(self):
        result = self.run_script("darkinfernotimedependent.py", "--help")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("heat_flow", result.stdout)
        self.assertIn("conductivity", result.stdout)
        self.assertIn("moon-inner-core", result.stdout)

    def test_radius_help_lists_lunar_scenarios(self):
        result = self.run_script("darkinfernoradius.py", "--help")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("moon-inner-core", result.stdout)

    def test_radius_rejects_non_finite_input_before_importing_asteria(self):
        result = self.run_script("darkinfernoradius.py", "nan", "-37")

        self.assertEqual(result.returncode, 2)
        self.assertIn("must be finite", result.stderr)
        self.assertNotIn("ModuleNotFoundError", result.stderr)

    def test_time_rejects_non_positive_physical_inputs(self):
        result = self.run_script("darkinfernotimedependent.py", "3", "0", "5500", "100")

        self.assertEqual(result.returncode, 2)
        self.assertIn("must be greater than zero", result.stderr)


if __name__ == "__main__":
    unittest.main()
