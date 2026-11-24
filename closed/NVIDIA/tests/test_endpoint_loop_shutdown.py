# SPDX-License-Identifier: Apache-2.0

"""Regression tests for joining the endpoint's daemon event loop at shutdown."""

import asyncio
import threading
import unittest
from unittest.mock import patch

from nv_mlpinf.llmlib.cores import trtllm_endpoint_core as endpoint


class EndpointLoopShutdownTest(unittest.TestCase):
    def setUp(self):
        self.globals = patch.multiple(endpoint, _module_loop=None,
                                      _loop_thread=None, _shutdown_event=None)
        self.globals.start()
        self.addCleanup(self.globals.stop)
        self.addCleanup(self.stop_loop)

    def stop_loop(self):
        if endpoint._shutdown_event is not None:
            endpoint._shutdown_event.set()
        if endpoint._loop_thread is not None:
            endpoint._loop_thread.join(timeout=15)

    def test_shutdown_without_initialization(self):
        endpoint._shutdown_module_loop()

    def test_shutdown_waits_for_pending_task_cleanup_and_loop_close(self):
        started = threading.Event()
        cancelled = threading.Event()
        endpoint._init_module_loop()

        async def pending_request():
            try:
                started.set()
                await asyncio.sleep(60)
            finally:
                cancelled.set()

        asyncio.run_coroutine_threadsafe(pending_request(), endpoint._module_loop)
        self.assertTrue(started.wait(timeout=5))
        endpoint._shutdown_module_loop()
        self.assertFalse(endpoint._loop_thread.is_alive())
        self.assertTrue(cancelled.is_set())
        self.assertTrue(endpoint._module_loop.is_closed())
        # atexit may call it again after explicit cleanup.
        endpoint._shutdown_module_loop()

    def test_shutdown_from_loop_thread_does_not_join_itself(self):
        done = threading.Event()
        errors = []
        endpoint._init_module_loop()

        def shutdown():
            try:
                endpoint._shutdown_module_loop()
            except Exception as error:
                errors.append(error)
            finally:
                done.set()

        endpoint._module_loop.call_soon_threadsafe(shutdown)
        self.assertTrue(done.wait(timeout=5))
        endpoint._loop_thread.join(timeout=15)
        self.assertFalse(endpoint._loop_thread.is_alive())
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
