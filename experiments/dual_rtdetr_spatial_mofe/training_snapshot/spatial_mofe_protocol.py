"""Fresh seed-zero spatial MoFE recipe and immutable experiment provenance."""
import copy
import json
from pathlib import Path

from dyfclt.runtime import load_config
from dyfclt.frequency_training import sha256, source_manifest

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = Path('/data/siwoo/DualRTDETR-PrePyramid-MoFE')
CONFIG = ROOT/'configs/rgbtdroneperson_spatial_mofe_seed0.yaml'
RUN = ROOT/'artifacts/training_launch_spatial_mofe_seed0_gpu1'
OUTPUT = ROOT/'outputs/rgbtdroneperson_spatial_mofe_seed0'
EXPECTED_UUID = 'GPU-c1faf7b9-2a34-010b-9739-4712a6f089a2'


def manifest(root=ROOT):
    root = Path(root)
    files = source_manifest(root)
    for path in sorted((root/'tests').rglob('*.py')):
        files[str(path.relative_to(root))] = sha256(path)
    for path in sorted((root/'docs').rglob('*.md')):
        files[str(path.relative_to(root))] = sha256(path)
    if (root/'README.md').exists(): files['README.md'] = sha256(root/'README.md')
    return files


def check_recipe(cfg):
    current = copy.deepcopy(cfg)
    previous = json.loads((ORIGINAL/'outputs/rgbtdroneperson_B_pre_seed0/config.json').read_text())
    frequency = current['model']['frequency_utility']
    if frequency.get('gate_mode') != 'spatial_mofe' or frequency.pop('residual_multiplier', None) != .5:
        raise ValueError('Expected spatial softmax and fixed residual multiplier 0.5')
    frequency['gate_mode'] = 'residual_mofe'
    for item in [current, previous]: item['train'].pop('output_dir', None)
    if current != previous:
        raise ValueError('All settings except spatial routing, multiplier and output must match B-pre')
    if cfg['train']['output_dir'] != str(OUTPUT.relative_to(ROOT)):
        raise ValueError('Unexpected output destination')
    return True


def verify_original():
    previous = json.loads((ROOT/'artifacts/original_source_frozen.json').read_text())
    if source_manifest(ORIGINAL) != previous:
        raise ValueError('Original B-pre/E-pre source changed')
    return len(previous)


def verify_tests():
    cert = json.loads((ROOT/'artifacts/test_certificate.json').read_text())
    if ((ROOT/'artifacts/full_tests_exit_code.txt').read_text().strip() != '0'
            or cert['source_sha256'] != manifest()
            or cert['log_sha256'] != sha256(ROOT/'artifacts/full_tests.log')):
        raise ValueError('CPU test certificate is failed or stale')
    return cert
