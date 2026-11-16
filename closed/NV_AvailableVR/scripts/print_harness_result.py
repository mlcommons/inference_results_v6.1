#! /usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Thin wrapper kept for backward compatibility with `make display_results`.
# Logic lives in nv_mlpinf.scripts.result_display.

from nv_mlpinf.scripts.result_display import print_session_results  # noqa: F401

from pathlib import Path
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print harness results")
    parser.add_argument("--log_dir", type=Path, help="Directory for all output logs",
                        default=os.environ.get("LOG_DIR", None))
    parser.add_argument("--ignore-accuracy-failure", default="0",
                        help="Ignore accuracy test failures and exit with status 0 (1 to enable, 0 to disable)")
    args = parser.parse_args()

    assert args.log_dir, "No log_dir specified"

    all_acc_pass = print_session_results(args.log_dir)

    if not all_acc_pass and args.ignore_accuracy_failure == "1":
        print("\nNote: Accuracy tests failed, but IGNORE_ACCURACY_FAILURE is set.")
    else:
        assert all_acc_pass, "Accuracy tests failed"
