import os
from pathlib import Path
import unittest
from unittest.mock import patch

from paths import (
    PATH_ENV_VAR,
    get_data_path,
    get_repository_path,
    get_visuals_path,
)


class RepositoryPathTests(unittest.TestCase):
    def test_configured_repository_paths(self):
        with patch.dict(os.environ, {PATH_ENV_VAR: "/tmp/darkinferno-test"}):
            root = Path("/tmp/darkinferno-test").resolve()
            self.assertEqual(get_repository_path(), root)
            self.assertEqual(get_data_path(), root / "data")
            self.assertEqual(get_visuals_path(), root / "visuals")

    def test_repository_path_defaults_to_checkout(self):
        with patch.dict(os.environ, {}, clear=True):
            root = Path(__file__).resolve().parents[1]
            self.assertEqual(get_repository_path(), root)
            self.assertEqual(os.environ[PATH_ENV_VAR], str(root))