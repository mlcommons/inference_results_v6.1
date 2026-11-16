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

# Standard library imports
import subprocess

# Third-party imports
from nvmitten.utils import run_command


class MPS:
    """Context manager for NVIDIA Multi-Process Service (MPS) control."""

    def __init__(self, active_sms: int = 100):
        """Initialize MPS controller with specified active SM percentage.

        Args:
            active_sms (int): Percentage of SMs to make active (1-100).
        """
        assert active_sms > 0 and active_sms <= 100
        self.active_sms = active_sms

    def is_enabled(self):
        """Check if MPS service is currently running.

        Returns:
            bool: True if MPS service is running, False otherwise.
        """
        cmd = "ps -ef | grep nvidia-cuda-mps-control | grep -c -v grep"
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        p.wait()
        output = p.stdout.readlines()
        return int(output[0]) >= 1

    def disable(self):
        """Stop the MPS service if it is running."""
        if self.is_enabled():
            cmd = "echo quit | nvidia-cuda-mps-control"
            run_command(cmd)

    def enable(self):
        """Start the MPS service with configured active SM percentage."""
        self.disable()
        if self.active_sms == 100:
            return

        cmd = f"export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE={self.active_sms} && nvidia-cuda-mps-control -d"
        run_command(cmd)

    def __enter__(self):
        """Enable MPS when entering context."""
        return self.enable()

    def __exit__(self, *args):
        """Disable MPS when exiting context."""
        self.disable()
