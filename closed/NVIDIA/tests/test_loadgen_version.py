# SPDX-License-Identifier: Apache-2.0

"""Regression tests for selecting the harness's submission-checker rules."""

import unittest
from unittest.mock import patch

from nv_mlpinf.common.mlcommons import loadgen


class ModelConfigVersionTest(unittest.TestCase):
    def setUp(self):
        self.configs = {"v6.0": {}, "v6.1": {}, "v6.2": {}}

    def test_installed_package_version_selects_matching_round(self):
        self.assertEqual(loadgen._get_model_config_version("6.1.0", self.configs), "v6.1")

    def test_prefixed_round_selects_matching_round(self):
        self.assertEqual(loadgen._get_model_config_version("v6.1", self.configs), "v6.1")

    def test_patch_and_prerelease_versions_keep_their_round(self):
        for version in ("6.1.2", "6.1.0rc1", "6.1.0.dev3"):
            with self.subTest(version=version):
                self.assertEqual(loadgen._get_model_config_version(version, self.configs), "v6.1")

    def test_matching_round_does_not_warn(self):
        with patch.object(loadgen.logging, "warning") as warning:
            loadgen._get_model_config_version("6.1.0", self.configs)
        warning.assert_not_called()

    def test_unknown_and_uninstalled_versions_fall_back_with_warning(self):
        for version in ("dev", "6.3.0"):
            with self.subTest(version=version), patch.object(loadgen.logging, "warning") as warning:
                self.assertEqual(loadgen._get_model_config_version(version, self.configs), "v6.2")
                warning.assert_called_once()

    def test_fallback_sorts_versions_numerically(self):
        self.assertEqual(loadgen._get_model_config_version("dev", {"v6.9": {}, "v6.10": {}}), "v6.10")


if __name__ == "__main__":
    unittest.main()
