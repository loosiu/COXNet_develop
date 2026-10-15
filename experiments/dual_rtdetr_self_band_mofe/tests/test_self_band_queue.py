import tempfile
from pathlib import Path
import unittest

from tools.staged_queue import GpuLocks

try:
    from tools.self_band_queue import assert_idle_predecessor_controller, reserve_resource_locks
except ImportError:
    assert_idle_predecessor_controller = reserve_resource_locks = None


class SelfBandQueueTests(unittest.TestCase):
    def state(self):
        return dict(current_stage=3, status='waiting_gpu', stages=[
            dict(stage=1,status='completed'), dict(stage=2,status='completed'), dict(stage=3,status='queued')])

    def test_retirement_requires_completed_second_stage(self):
        self.assertIsNotNone(assert_idle_predecessor_controller)
        state=self.state();state['stages'][1]['status']='running'
        with self.assertRaises(ValueError):assert_idle_predecessor_controller(state)

    def test_started_third_stage_must_never_be_replaced(self):
        self.assertIsNotNone(assert_idle_predecessor_controller)
        for change in ('status','pid'):
            state=self.state()
            state['stages'][2][change]='running' if change=='status' else 999
            with self.assertRaises(ValueError):assert_idle_predecessor_controller(state)

    def test_only_idle_queue_can_be_retired(self):
        self.assertIsNotNone(assert_idle_predecessor_controller)
        self.assertTrue(assert_idle_predecessor_controller(self.state()))
        state=self.state();state['status']='failed'
        with self.assertRaises(ValueError):assert_idle_predecessor_controller(state)

    def test_reservation_keeps_all_resource_locks(self):
        self.assertIsNotNone(reserve_resource_locks)
        with tempfile.TemporaryDirectory() as directory:
            paths=[Path(directory)/f'{i}.lock' for i in range(3)]
            locks=reserve_resource_locks(paths)
            try:
                probe=GpuLocks(paths);self.assertFalse(probe.acquire())
                self.assertEqual(len(locks.handles),3)
            finally:locks.release()
            self.assertTrue(probe.acquire());probe.release()


if __name__=='__main__':unittest.main()
