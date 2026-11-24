# SPDX-License-Identifier: Apache-2.0

"""Regression tests for concurrent cleanup of audit artifacts."""

import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from nv_mlpinf.common.mlcommons import compliance


class AuditCleanupTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        previous = Path.cwd()
        os.chdir(self.temp.name)
        self.addCleanup(os.chdir, previous)

    def test_missing_artifacts_are_allowed(self):
        compliance.set_audit_conf()

    def test_concurrent_removal_does_not_abort_remaining_cleanup(self):
        artifacts = ["audit.config", "verify_accuracy.txt", "verify_performance.txt",
                     "mlperf_log_accuracy_baseline.json", "accuracy.txt", "predictions.json"]
        for name in artifacts:
            Path(name).write_text("old result")
        remove = os.remove

        def remove_after_other_rank(name):
            remove(name)
            # The other rank removed it between exists() and remove().
            remove(name)

        with patch.object(compliance.os, "remove", side_effect=remove_after_other_rank):
            compliance.set_audit_conf()
        self.assertFalse(any(Path(name).exists() for name in artifacts))

    def test_permission_errors_are_not_hidden(self):
        Path("audit.config").write_text("old config")
        with patch.object(compliance.os, "remove", side_effect=PermissionError("denied")):
            with self.assertRaises(PermissionError):
                compliance.set_audit_conf()

    def test_new_audit_config_is_copied_after_concurrent_cleanup(self):
        root = Path(self.temp.name) / "reference"
        source = root / "compliance" / "TEST01" / "benchmark" / "audit.config"
        source.parent.mkdir(parents=True)
        source.write_text("new config")
        Path("audit.config").write_text("old config")
        remove = os.remove

        def remove_after_other_rank(name):
            remove(name)
            remove(name)

        with patch.object(compliance.paths, "MLCOMMONS_INF_REPO", root), \
                patch.object(compliance.os, "remove", side_effect=remove_after_other_rank):
            compliance.set_audit_conf(SimpleNamespace(valstr="TEST01"),
                                      SimpleNamespace(valstr="benchmark"))
        self.assertEqual(Path("audit.config").read_text(), "new config")


if __name__ == "__main__":
    unittest.main()
