import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

try:
    from tools.staged_queue import verify_completed_run, GpuLocks
except ImportError:
    verify_completed_run = GpuLocks = None


class StagedQueueTests(unittest.TestCase):
    def create_run(self, root, complete=True):
        output, run = root/'output', root/'run'
        output.mkdir(); run.mkdir()
        cfg = dict(seed=0, train=dict(epochs=20, batch_size=2, ema=dict(enabled=True,decay=.9999,warmups=2000)),
                   data={}, model={}, evaluation=dict(selection_metric='AP50'))
        rows = [dict(epoch=e, evaluation_weights='ema', ema_updates=e*2450,
                     validation=dict(AP50=.9 if e == 3 else .1, images=1225, partial=False))
                for e in range(1, 21 if complete else 2)]
        (run/'train.log').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        (run/'exit_code.txt').write_text('0\n')
        (output/'config.json').write_text(json.dumps(cfg))
        for name, epoch in [('last.pth', 19), ('best.pth', 2)]:
            state = dict(epoch=epoch, global_step=(epoch+1)*2450, best_ap=.9, partial_epoch=False,
                         config=cfg, category_ids=[0,1,2], model={'weight':torch.ones(1)},
                         evaluation_weights='ema',
                         ema={'module':{'weight':torch.ones(1)},'updates':(epoch+1)*2450,'decay':.9999,'warmups':2000})
            torch.save(state, output/name)
        return output, run, cfg

    def require_api(self):
        self.assertIsNotNone(verify_completed_run, 'completion gate for queued training is not implemented')

    def test_no_exit_file_waits_without_launching_next_stage(self):
        self.require_api()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root/'run').mkdir()
            self.assertIsNone(verify_completed_run(root/'output', root/'run'))

    def test_success_exit_without_twenty_validations_cannot_release_queue(self):
        self.require_api()
        with tempfile.TemporaryDirectory() as directory:
            output, run, cfg = self.create_run(Path(directory), complete=False)
            with self.assertRaises(ValueError): verify_completed_run(output, run, cfg)

    def test_failed_training_never_releases_next_stage(self):
        self.require_api()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run=root/'run';run.mkdir();(run/'exit_code.txt').write_text('143\n')
            with self.assertRaises(RuntimeError):verify_completed_run(root/'output',run)

    def test_best_epoch_is_distinct_from_final_and_checkpoints_are_required(self):
        self.require_api()
        with tempfile.TemporaryDirectory() as directory:
            output, run, cfg = self.create_run(Path(directory))
            result=verify_completed_run(output,run,cfg)
            self.assertEqual(result['final_epoch'],20);self.assertEqual(result['best_epoch'],3)
            self.assertEqual(result['global_step'],49000)
            (output/'best.pth').unlink()
            with self.assertRaises(ValueError):verify_completed_run(output,run,cfg)

    def test_checkpoint_with_nonfinite_weights_is_not_complete(self):
        self.require_api()
        with tempfile.TemporaryDirectory() as directory:
            output, run, cfg = self.create_run(Path(directory))
            d=torch.load(output/'best.pth',map_location='cpu',weights_only=False)
            d['ema']['module']['weight'].fill_(float('nan'));torch.save(d,output/'best.pth')
            with self.assertRaises(ValueError):verify_completed_run(output,run,cfg)

    def test_gpu_lock_conflict_releases_partial_acquisition(self):
        self.assertIsNotNone(GpuLocks, 'GPU queue exclusion is not implemented')
        with tempfile.TemporaryDirectory() as directory:
            paths=[Path(directory)/name for name in ('a.lock','b.lock')]
            owner=GpuLocks([paths[1]]);self.assertTrue(owner.acquire())
            second=GpuLocks(paths);self.assertFalse(second.acquire())
            probe=GpuLocks([paths[0]]);self.assertTrue(probe.acquire());probe.release()
            owner.release();self.assertTrue(second.acquire());second.release()

    def test_cuda_mapping_probe_refuses_other_physical_gpu(self):
        from tools import staged_queue
        self.assertTrue(hasattr(staged_queue,'probe_cuda_mapping'))
        with patch.object(staged_queue.subprocess,'check_output',return_value=json.dumps(
                dict(gpu_uuid='unexpected-gpu', device='cuda:0'))):
            with self.assertRaises(RuntimeError):staged_queue.probe_cuda_mapping({},'requested-gpu')

    def test_checkpoint_completion_requires_finite_best_metric_and_correct_ema_metadata(self):
        for key,value in [('best_ap',float('nan')),('evaluation_weights','model'),
                          ('updates',0),('decay',.5),('warmups',100)]:
            with self.subTest(key=key),tempfile.TemporaryDirectory() as directory:
                output,run,cfg=self.create_run(Path(directory))
                state=torch.load(output/'best.pth',map_location='cpu',weights_only=False)
                if key in ('updates','decay','warmups'):
                    state['ema'][key]=value
                else:
                    state[key]=value
                torch.save(state,output/'best.pth')
                with self.assertRaises(ValueError):verify_completed_run(output,run,cfg)


if __name__=='__main__':unittest.main()
