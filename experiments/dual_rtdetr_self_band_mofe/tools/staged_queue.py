"""Wait for verified 20-epoch completion, then run three independent GPU1 experiments."""
import argparse
import fcntl
import gc
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from dyfclt.frequency_training import sha256, source_manifest, validate_output_destination, write_launch_provenance
from dyfclt.runtime import load_config


def atomic_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


class GpuLocks:
    def __init__(self, paths):
        self.paths, self.handles = list(paths), []

    def acquire(self):
        try:
            for path in self.paths:
                handle = Path(path).open('a')
                self.handles.append(handle)
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            self.release()
            return False

    def release(self):
        for handle in reversed(self.handles):
            handle.close()
        self.handles = []


def verify_completed_run(output, run, expected_config=None):
    output, run = Path(output), Path(run)
    if not (run/'exit_code.txt').exists():
        return None
    code = int((run/'exit_code.txt').read_text().strip())
    if code != 0:
        raise RuntimeError('Preceding training failed with exit code ' + str(code))
    cfg = json.loads((output/'config.json').read_text())
    if expected_config is not None and cfg != expected_config:
        raise ValueError('Completed run config differs from its authorized config')
    if cfg['train']['epochs'] != 20 or cfg['train']['batch_size'] != 2 or cfg['seed'] != 0:
        raise ValueError('Expected fresh seed0/batch2/20 epochs')
    if not cfg['train'].get('ema', {}).get('enabled'):
        raise ValueError('Expected EMA-enabled training')
    validations = {}
    for line in (run/'train.log').read_text().splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if not isinstance(row, dict) or 'validation' not in row:
            continue
        epoch, metrics = row['epoch'], row['validation']
        if epoch in validations or row.get('evaluation_weights') != 'ema' or row.get('ema_updates') != epoch * 2450:
            raise ValueError('Duplicate epoch or incomplete EMA validation')
        if metrics.get('partial') is not False or metrics.get('images') != 1225 or not 0 <= metrics['AP50'] <= 1:
            raise ValueError('Full finite validation required')
        validations[epoch] = metrics
    if set(validations) != set(range(1, 21)):
        raise ValueError('Twenty completed validations required')
    best_epoch = max(validations, key=lambda e: validations[e]['AP50'])
    best_ap = validations[best_epoch]['AP50']
    result = dict(final_epoch=20, best_epoch=best_epoch, best_AP50=best_ap, global_step=49000,
                  exit_code=code, output=str(output), run=str(run))
    for name in ('last.pth', 'best.pth'):
        path = output/name
        if not path.is_file():
            raise ValueError('Required checkpoint missing: ' + str(path))
        state = torch.load(path, map_location='cpu', weights_only=False)
        expected_epoch = 20 if name == 'last.pth' else best_epoch
        ema_options = cfg['train']['ema']
        if (state.get('partial_epoch') or state['epoch'] + 1 != expected_epoch
                or state['global_step'] != expected_epoch * 2450 or state['config'] != cfg
                or not math.isfinite(state['best_ap']) or abs(state['best_ap'] - best_ap) > 1e-12
                or state.get('category_ids') != [0, 1, 2] or state.get('evaluation_weights') != 'ema'
                or not state.get('ema') or not state.get('model')
                or state['ema'].get('updates') != expected_epoch * 2450
                or state['ema'].get('decay') != ema_options.get('decay', .9999)
                or state['ema'].get('warmups') != ema_options.get('warmups', 2000)):
            raise ValueError('Checkpoint completion metadata invalid: ' + name)
        for dictionary in (state['model'], state['ema']['module']):
            if not all(not value.is_floating_point() or bool(torch.isfinite(value).all()) for value in dictionary.values()):
                raise ValueError('Nonfinite checkpoint weights: ' + name)
        del state
        gc.collect()
    return result


def gpu_available(uuid):
    actual = subprocess.check_output(['nvidia-smi', '-i', '1', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    if actual != uuid:
        raise RuntimeError('Physical GPU1 UUID changed')
    jobs = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader'], text=True)
    return uuid not in jobs


def probe_cuda_mapping(env, expected_uuid):
    """Verify numeric CUDA selection with one tensor before allocating a model."""
    probe = '''import json,os,subprocess,torch
assert torch.cuda.device_count()==1
tensor=torch.zeros(1,device='cuda:0')
rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True).splitlines()
actual=[row.split(',')[1].strip() for row in rows if int(row.split(',')[0])==os.getpid()]
assert actual==[os.environ['STAGED_EXPECTED_GPU_UUID']],actual
print(json.dumps(dict(gpu_uuid=actual[0],device='cuda:0',mapping_verified=True)))
'''
    result = json.loads(subprocess.check_output([sys.executable, '-c', probe],
                         env=dict(env, STAGED_EXPECTED_GPU_UUID=expected_uuid), text=True))
    if result.get('gpu_uuid') != expected_uuid or result.get('device') != 'cuda:0':
        raise RuntimeError('CUDA mapping probe does not match requested physical GPU')
    return result


def validate_certificate(plan):
    certificate = json.loads((ROOT/'artifacts/training_certificate.json').read_text())
    if not certificate.get('ready') or certificate['source_sha256'] != source_manifest(ROOT):
        raise ValueError('Training source changed or validation certificate missing')
    if certificate['plan_sha256'] != sha256(ROOT/'artifacts/experiment_plan.json'):
        raise ValueError('Queue plan changed after verification')
    for path, digest in certificate['original_source_sha256'].items():
        if sha256(path) != digest:
            raise ValueError('Preserved original source changed: ' + path)
    for path, digest in certificate['input_sha256'].items():
        if sha256(path) != digest:
            raise ValueError('Data/pretrained changed: ' + path)
    for path, digest in certificate['verification_sha256'].items():
        if sha256(ROOT/path) != digest:
            raise ValueError('Verification artifact changed: ' + path)
    for entry in plan['stages']:
        if load_config(ROOT/entry['config']) != certificate['configs'][str(entry['stage'])]:
            raise ValueError('Stage config changed')
    return certificate


def interrupted(signum, frame):
    raise SystemExit(128 + signum)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--poll-seconds', type=float, default=30)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error('--poll-seconds must be positive')
    plan = json.loads((ROOT/'artifacts/experiment_plan.json').read_text())
    certificate = validate_certificate(plan)
    qdir = ROOT/'artifacts/queue_gpu1';qdir.mkdir(exist_ok=True)
    queue_lock = GpuLocks([Path('/tmp/mofe_staged_queue_gpu1.lock')])
    if not queue_lock.acquire():
        raise RuntimeError('Another staged queue already owns this queue')
    signal.signal(signal.SIGTERM, interrupted);signal.signal(signal.SIGINT, interrupted)
    (qdir/'pid.txt').write_text(str(os.getpid())+'\n')
    state = dict(status='waiting_predecessor', controller_pid=os.getpid(), physical_gpu=1,
                 gpu_uuid=plan['gpu_uuid'], stages=[dict(entry, status='queued') for entry in plan['stages']],
                 pretrained_only=True, trained_checkpoint_loaded=False)
    child = None; active_run = None; gpu_locks = None

    def publish():
        state['updated_at_utc'] = datetime.now(timezone.utc).isoformat()
        atomic_json(qdir/'status.json', state)

    try:
        publish()
        predecessor = verify_completed_run(plan['predecessor_output'], plan['predecessor_run'], certificate['predecessor_config'])
        if args.check_only:
            print(json.dumps(dict(queue_valid=True, predecessor_complete=predecessor is not None)), flush=True)
            return
        while predecessor is None:
            time.sleep(args.poll_seconds)
            predecessor = verify_completed_run(plan['predecessor_output'], plan['predecessor_run'], certificate['predecessor_config'])
            publish()
        state['predecessor_completion'] = predecessor
        for index, entry in enumerate(plan['stages']):
            cfg = load_config(ROOT/entry['config']);run = ROOT/entry['run_dir'];output = ROOT/entry['output_dir']
            if (run/'exit_code.txt').exists():
                state['stages'][index].update(status='completed', completion=verify_completed_run(output, run, cfg))
                publish();continue
            validate_output_destination(output)
            if (run/'launch.json').exists():
                raise ValueError('Existing attempt has no verified completion; do not restart over it')
            # Resource locks are distinct from queue-controller locks. Never take
            # another waiting queue's controller lock to reserve the GPU.
            paths = [Path('/tmp')/name for name in sorted((
                'cnn_mofe_zero_gpu1.lock', 'dyfclt_gpu1.lock', 'spatial_mofe_gpu1.lock',
                'pre_pyramid_mofe_gpu1.lock', 'yolox_auair_s_1280_gpu1.lock',
                'coxnet_icbfc_gpu1.lock', 'trpc_gpu1.lock',
                'coxnet_dwt_dfca_gpu1.lock', 'coxnet_oepc_gpu1.lock'))]
            gpu_locks = GpuLocks(paths)
            state.update(status='waiting_gpu', current_stage=entry['stage']);publish()
            while True:
                if gpu_locks.acquire():
                    if gpu_available(plan['gpu_uuid']):
                        break
                    gpu_locks.release()
                time.sleep(args.poll_seconds);publish()
            cert = validate_certificate(plan)
            run.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES='1',
                       OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONDONTWRITEBYTECODE='1',
                       MPLCONFIGDIR='/tmp/staged_mofe_matplotlib')
            mapping = probe_cuda_mapping(env, plan['gpu_uuid'])
            atomic_json(run/'physical_gpu_mapping.json', mapping)
            command = [sys.executable, '-u', 'tools/train.py', '--config', entry['config'], '--device', 'cuda:0']
            assert '--resume' not in command and '--no-pretrained' not in command
            launch = dict(started_utc=datetime.now(timezone.utc).isoformat(), stage=entry['stage'],
                command=command, working_directory=str(ROOT), resolved_config=cfg, expected_total_updates=49000,
                physical_gpu=1, gpu_uuid=plan['gpu_uuid'], device='cuda:0', source_sha256=cert['source_sha256'],
                initialization='fresh public backbone pretrained', trained_A_B_previous_stage_checkpoint_loaded=False,
                preflight_updated_weights_used_for_production=False, evaluation_weights='ema', selection_metric='AP50',
                output_directory=str(output), locks=[str(p) for p in paths])
            write_launch_provenance(run, ROOT, cert['source_sha256'], cfg, launch)
            active_run = run
            with (run/'train.log').open('x') as log:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                         pass_fds=tuple(h.fileno() for h in gpu_locks.handles))
                (run/'pid.txt').write_text(str(child.pid)+'\n')
                state['stages'][index].update(status='running', pid=child.pid)
                state.update(status='running', current_stage=entry['stage']);publish()
                print(json.dumps(dict(stage=entry['stage'], pid=child.pid, config=entry['config'], output=str(output))), flush=True)
                while child.poll() is None:
                    time.sleep(args.poll_seconds)
                    jobs = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader'], text=True)
                    actual = [line.split(',')[1].strip() for line in jobs.splitlines() if int(line.split(',')[0]) == child.pid]
                    if actual:
                        if actual != [plan['gpu_uuid']]:
                            raise RuntimeError('Training process mapped to unexpected physical GPU')
                        state['stages'][index]['physical_mapping_verified'] = True
                    publish()
            (run/'exit_code.txt').write_text(str(child.returncode)+'\n')
            gpu_locks.release();gpu_locks = None;child = None;active_run = None
            completion = verify_completed_run(output, run, cfg)
            state['stages'][index].update(status='completed', completion=completion)
            publish()
        state['status'] = 'completed';publish()
        (qdir/'exit_code.txt').write_text('0\n')
    except BaseException as error:
        state.update(status='stopped' if isinstance(error, (SystemExit, KeyboardInterrupt)) else 'failed', error=str(error))
        publish()
        (qdir/'exit_code.txt').write_text(str(error.code if isinstance(error, SystemExit) else 1)+'\n')
        raise
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill();child.wait()
            if active_run is not None:
                (active_run/'exit_code.txt').write_text(str(child.returncode)+'\n')
        if gpu_locks is not None:
            gpu_locks.release()
        queue_lock.release()


if __name__ == '__main__':
    main()
