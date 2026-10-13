"""Freeze the checked configuration and source before a fresh production run."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from dyfclt.runtime import load_config
from dyfclt.frequency_training import sha256, validate_output_destination, write_launch_provenance
from tools.spatial_mofe_protocol import CONFIG, RUN, OUTPUT, EXPECTED_UUID, manifest, check_recipe, verify_tests, verify_original


def main():
    cfg = load_config(CONFIG); check_recipe(cfg); verify_tests(); verify_original()
    proof = json.loads((ROOT/'artifacts/verification_spatial_gpu1.json').read_text())
    assert proof['resolved_config'] == cfg and proof['source_sha256'] == manifest()
    assert os.environ['RESIDUAL_GPU_UUID'] == proof['gpu_uuid'] == EXPECTED_UUID
    for key in ['all_B_pre_initial_tensors_equal', 'CPU_rng_equal', 'cpu_tests_passed',
                'all_module_gradients_finite_nonzero', 'all_module_parameters_updated',
                'optimizer_includes_all_module_parameters', 'EMA_eval_finite_repeatable', 'spatial_weights_verified']:
        assert proof[key], key
    assert sha256(ROOT/cfg['model']['pretrained']) == proof['pretrained_sha256']
    for split in ['train', 'val']: assert sha256(cfg['data'][split]['ann_file']) == proof['input_sha256'][split]
    validate_output_destination(OUTPUT); RUN.mkdir(parents=True, exist_ok=True)
    assert not (RUN/'launch.json').exists(), 'Preserve previous launch'
    launch = dict(started_utc=datetime.now(timezone.utc).isoformat(), revision='spatial_mofe_v1',
        mode='fresh_seed0_spatial_softmax_P3_before_FPN_residual_multiplier_0.5', resolved_config=cfg,
        expected_total_updates=49000, physical_gpu=1, gpu_uuid=EXPECTED_UUID, device='cuda:0',
        parameters=proof['parameters'], module_parameters=proof['module_parameters'],
        source_sha256=proof['source_sha256'], input_sha256=proof['input_sha256'], pretrained_sha256=proof['pretrained_sha256'],
        trained_A_B_E_checkpoint_loaded=False, preflight_updated_weights_used_for_production=False,
        verification='artifacts/verification_spatial_gpu1.json',
        mechanism='Original bidirectional four convolution experts; per-cell four-way softmax; original + 0.5*learned_gamma*update; projected P3 then original AIFI/FPN/PAN/fusion/decoder; detection loss only',
        interpretation='Spatial routing and residual multiplier both differ from B-pre; their independent contribution requires a separate ablation. Learned gamma/expert growth can compensate fixed multiplier.')
    write_launch_provenance(RUN, ROOT, proof['source_sha256'], cfg, launch)
    print(json.dumps(dict(launch=str(RUN/'launch.json'))), flush=True)


if __name__ == '__main__': main()
