import json
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
import torch
from dyfclt.runtime import warmup_factor, postprocess, save_checkpoint, load_checkpoint, resize_training_batch, evaluate, select_training_config, apply_optimizer_step


class RuntimeTests(unittest.TestCase):
    def test_warmup_exact_endpoints_and_constant_tail(self):
        self.assertEqual(warmup_factor(0, 2000, .001), .001)
        self.assertAlmostEqual(warmup_factor(1000, 2000, .001), .5005)
        self.assertEqual(warmup_factor(2000, 2000, .001), 1.)
        self.assertEqual(warmup_factor(4000, 2000, .001), 1.)
        self.assertEqual(warmup_factor(0, 0, .001), 1.)

    def test_postprocess_restores_non_square_original_coordinates(self):
        outputs = {'pred_logits': torch.tensor([[[10., -10.], [-10., 9.]]]),
                   'pred_boxes': torch.tensor([[[.5, .5, .2, .4], [.2, .25, .1, .1]]])}
        targets = [{'orig_size': torch.tensor([100, 200]), 'image_id': torch.tensor(7)}]
        records = postprocess(outputs, targets, [3, 8], topk=2)
        self.assertEqual([r['category_id'] for r in records], [3, 8])
        self.assertEqual(records[0]['image_id'], 7)
        torch.testing.assert_close(torch.tensor(records[0]['bbox']), torch.tensor([80.,30.,40.,40.]))

    def test_random_batch_resize_preserves_modal_alignment_and_normalized_boxes(self):
        rgb = torch.arange(3*8*10).view(1,3,8,10).float()
        targets = [{'boxes': torch.tensor([[.5,.5,.2,.4]]), 'size': torch.tensor([8,10])}]
        before = targets[0]['boxes'].clone()
        resized_rgb, resized_ir, resized_targets = resize_training_batch(rgb, rgb.clone(), targets, [[32,64]])
        self.assertEqual(resized_rgb.shape, (1,3,32,64))
        torch.testing.assert_close(resized_rgb, resized_ir)
        torch.testing.assert_close(resized_targets[0]['boxes'], before)
        self.assertEqual(resized_targets[0]['size'].tolist(), [32,64])

    def test_evaluation_uses_visited_images_and_restores_training_mode(self):
        class FixedDetector(torch.nn.Module):
            def forward(self, rgb, thermal):
                return {'pred_logits': torch.tensor([[[15.]]]),
                        'pred_boxes': torch.tensor([[[.5,.5,.2,.4]]])}
        model = FixedDetector().train()
        loader = [(torch.zeros(1,3,32,32), torch.zeros(1,3,32,32),
                   [{'orig_size': torch.tensor([100,200]), 'image_id': torch.tensor(4)}])]
        ground_truth = {'images': [{'id':4,'width':200,'height':100}, {'id':9,'width':200,'height':100}],
                        'categories':[{'id':3,'name':'person'}],
                        'annotations':[{'id':1,'image_id':4,'category_id':3,'bbox':[80,30,40,40],'area':1600,'iscrowd':0},
                                       {'id':2,'image_id':9,'category_id':3,'bbox':[80,30,40,40],'area':1600,'iscrowd':0}]}
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            annotation = Path(tmp)/'ground_truth.json'
            annotation.write_text(json.dumps(ground_truth))
            metrics = evaluate(model, loader, 'cpu', annotation, [3], Path(tmp)/'eval', max_batches=1)
            self.assertAlmostEqual(metrics['AP'], 1., places=5)
            self.assertEqual(metrics['images'], 1)
            self.assertTrue(metrics['partial'])
            self.assertTrue(model.training)
            self.assertEqual(json.loads(annotation.read_text()), ground_truth)

    def test_resume_uses_saved_configuration_and_explicit_config_can_override(self):
        saved = {'model': {'num_classes': 2, 'hidden_dim': 32}, 'train': {'epochs': 5}}
        chosen = select_training_config(saved_config=saved)
        self.assertEqual(chosen, saved)
        chosen['model']['hidden_dim'] = 64
        self.assertEqual(saved['model']['hidden_dim'], 32)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'override.yaml'
            path.write_text('model:\n  num_classes: 4\ntrain:\n  epochs: 20\n')
            override = select_training_config(path, saved)
            self.assertEqual(override['model']['num_classes'], 4)
            self.assertEqual(override['train']['epochs'], 20)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA GradScaler test')
    def test_amp_overflow_skips_update_then_recovers_with_finite_gradient(self):
        model = torch.nn.Linear(1, 1, bias=False).cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        scaler = torch.cuda.amp.GradScaler()
        from dyfclt.training import ModelEMA
        ema = ModelEMA(model)
        before = model.weight.detach().clone()
        scaler.scale(model.weight.sum() * 1e38).backward()
        norm, applied = apply_optimizer_step(model, optimizer, scaler, .1, ema=ema)
        self.assertFalse(applied)
        self.assertEqual(ema.updates, 0)
        torch.testing.assert_close(ema.module.weight, before)
        torch.testing.assert_close(model.weight, before)
        self.assertEqual(scaler.get_scale(), 32768.)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(model.weight.sum()).backward()
        norm, applied = apply_optimizer_step(model, optimizer, scaler, .1, ema=ema)
        self.assertTrue(applied)
        self.assertEqual(ema.updates, 1)
        self.assertFalse(torch.equal(ema.module.weight, before))
        self.assertTrue(torch.isfinite(norm))
        self.assertFalse(torch.equal(model.weight, before))

    def test_checkpoint_roundtrip_restores_optimizer_and_metadata(self):
        torch.manual_seed(17)
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        model(torch.ones(1,2)).sum().backward()
        optimizer.step()
        before = model.weight.detach().clone()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'last.pth'
            save_checkpoint(path, model, optimizer, None, {'model': {}}, 2, 31, .12, [3,8])
            with torch.no_grad():
                model.weight.zero_()
            ckpt = load_checkpoint(path, model, optimizer)
            torch.testing.assert_close(model.weight, before)
            self.assertEqual(ckpt['epoch'], 2)
            self.assertEqual(ckpt['global_step'], 31)
            self.assertEqual(ckpt['category_ids'], [3,8])
            self.assertFalse(Path(str(path) + '.tmp').exists())

if __name__ == '__main__':
    unittest.main()
