"""Reserve GPU1 after stage2, retire its idle queue, then run self-band MoFE."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
OLD = Path('/data/siwoo/DualRTDETR-MoFE-Staged')
sys.path.insert(0, str(ROOT))
from dyfclt.frequency_training import sha256, source_manifest, validate_output_destination, write_launch_provenance
from dyfclt.runtime import load_config
from tools.staged_queue import GpuLocks, atomic_json, gpu_available, probe_cuda_mapping, verify_completed_run

GPU_UUID = 'GPU-c1faf7b9-2a34-010b-9739-4712a6f089a2'
CONFIG = 'configs/rgbtdroneperson_self_band_mofe_zero_post_encoder_seed0.yaml'
OLD_RUN = OLD/'artifacts/training_launch_stage2_seed0_gpu1'
OLD_OUTPUT = OLD/'outputs/rgbtdroneperson_donor_mofe_zero_post_encoder_seed0'
OLD_THIRD = OLD/'artifacts/training_launch_stage3_seed0_gpu1'
OLD_STATUS = OLD/'artifacts/queue_gpu1/status.json'
RUN = ROOT/'artifacts/training_launch_self_band_seed0_gpu1'
QDIR = ROOT/'artifacts/queue_self_band'


def assert_idle_predecessor_controller(state):
    by_stage = {entry['stage']: entry for entry in state['stages']}
    if by_stage[2]['status'] != 'completed':
        raise ValueError('Cannot retire a controller with an unfinished stage2')
    third = by_stage[3]
    if third.get('pid') is not None or third['status'] != 'queued':
        raise ValueError('Stage3 already started; preserve it and refuse replacement')
    if state['status'] != 'waiting_gpu' or state['current_stage'] != 3:
        raise ValueError('Controller must be idle waiting for the cancelled third stage')
    return True


def reserve_resource_locks(paths):
    """Blocking first lock waits for stage2's explicit release, not GPU polling."""
    locks = GpuLocks(sorted(map(Path, paths)))
    try:
        for index, path in enumerate(locks.paths):
            handle = path.open('a')
            locks.handles.append(handle)
            fcntl.flock(handle, fcntl.LOCK_EX | (fcntl.LOCK_NB if index else 0))
        return locks
    except BaseException:
        locks.release()
        raise


def interrupt(signum, frame):
    raise SystemExit(128+signum)


def verify_certificate():
    cert = json.loads((ROOT/'artifacts/training_certificate.json').read_text())
    if not cert.get('ready') or cert['source_sha256'] != source_manifest(ROOT):
        raise ValueError('Self-band sources differ from their verified certificate')
    for path, expected in cert['input_sha256'].items():
        if sha256(path) != expected:
            raise ValueError('Verified input changed: '+path)
    for path, expected in cert['verification_sha256'].items():
        if sha256(ROOT/path) != expected:
            raise ValueError('Verification artifact changed: '+path)
    old_hashes = json.loads((ROOT/'artifacts/preserved_staged_source_sha256.json').read_text())
    if source_manifest(OLD) != old_hashes:
        raise ValueError('Preserved stage2 source files changed')
    cfg = load_config(ROOT/CONFIG)
    if cfg != cert['config']:
        raise ValueError('Certified config changed')
    return cert, cfg


def main():
    QDIR.mkdir(parents=True, exist_ok=True)
    controller_lock = GpuLocks([Path('/tmp/self_band_mofe_queue_gpu1.lock')])
    if not controller_lock.acquire():
        raise RuntimeError('Self-band replacement supervisor already exists')
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    (QDIR/'pid.txt').write_text(str(os.getpid())+'\n')
    state = dict(status='checking_predecessor', controller_pid=os.getpid(), physical_gpu=1,
                 gpu_uuid=GPU_UUID, replaces='unstarted donor_strength stage3',
                 config=CONFIG, predecessor_pid=3127815, stage='replacement_stage3',
                 trained_previous_checkpoint_loaded=False)
    locks = None
    child = None

    def publish(**values):
        state.update(values, updated_at_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(QDIR/'status.json', state)

    try:
        old_state = json.loads(OLD_STATUS.read_text())
        third = next(s for s in old_state['stages'] if s['stage'] == 3)
        if third['status'] != 'queued' or third.get('pid') is not None or (OLD_THIRD/'launch.json').exists():
            raise ValueError('Third experiment already started; replacement is not authorized')
        old_pid = old_state['controller_pid']
        old_start = Path(f'/proc/{old_pid}/stat').read_text().split()[21]
        if Path(f'/proc/{old_pid}/cwd').resolve() != OLD:
            raise ValueError('Unexpected predecessor controller working directory')
        launch = json.loads((OLD_RUN/'launch.json').read_text())
        paths = launch['locks']
        publish(status='waiting_stage2_resource_release', old_controller_pid=old_pid,
                old_controller_start_ticks=old_start, locks=paths)
        # While stage2 runs, its parent and child retain these same locks. The
        # blocked waiter obtains them after that parent writes the real exit code.
        locks = reserve_resource_locks(paths)
        publish(status='verifying_stage2_completion')
        cfg2 = json.loads((OLD_OUTPUT/'config.json').read_text())
        completion = verify_completed_run(OLD_OUTPUT, OLD_RUN, cfg2)
        if completion is None:
            raise ValueError('GPU locks released without a stage2 completion record')
        deadline = time.monotonic()+300
        while True:
            old_state = json.loads(OLD_STATUS.read_text())
            if old_state['current_stage'] == 3 and old_state['status'] == 'waiting_gpu':
                assert_idle_predecessor_controller(old_state)
                break
            if old_state['status'] in ('failed', 'stopped', 'completed') or time.monotonic() > deadline:
                raise RuntimeError('Predecessor queue did not reach a safe idle handoff')
            time.sleep(1)
        if (OLD_THIRD/'launch.json').exists():
            raise ValueError('Original third launch unexpectedly exists')
        if Path(f'/proc/{old_pid}/stat').read_text().split()[21] != old_start:
            raise ValueError('Predecessor controller PID was reused')
        command = Path(f'/proc/{old_pid}/cmdline').read_bytes().replace(b'\0', b' ')
        if b'tools/staged_queue.py' not in command:
            raise ValueError('Unexpected predecessor controller process')
        # No active child exists here: stage2 is verified, stage3 is lock-blocked.
        os.kill(old_pid, signal.SIGTERM)
        for _ in range(60):
            if (OLD/'artifacts/queue_gpu1/exit_code.txt').exists():
                break
            time.sleep(1)
        else:
            raise RuntimeError('Old controller has not acknowledged retirement')
        publish(status='waiting_verified_preflight', predecessor_completion=completion,
                old_third_cancelled_before_start=True, old_controller_retired=True)
        while not (ROOT/'artifacts/training_certificate.json').exists():
            time.sleep(5)
        cert, cfg = verify_certificate()
        output = ROOT/cfg['train']['output_dir']
        validate_output_destination(output)
        if RUN.exists():
            raise FileExistsError('Refuse to overwrite an existing self-band attempt')
        if not gpu_available(GPU_UUID):
            raise RuntimeError('Unexpected GPU1 occupant after resource reservation')
        RUN.mkdir(parents=True)
        env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES='1',
                   OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONDONTWRITEBYTECODE='1',
                   MPLCONFIGDIR='/tmp/self_band_mofe_matplotlib')
        mapping = probe_cuda_mapping(env, GPU_UUID)
        atomic_json(RUN/'physical_gpu_mapping.json', mapping)
        command = [sys.executable, '-u', 'tools/train.py', '--config', CONFIG, '--device', 'cuda:0']
        info = dict(command=command, working_directory=str(ROOT), resolved_config=cfg,
                    expected_total_updates=49000, physical_gpu=1, gpu_uuid=GPU_UUID,
                    device='cuda:0', source_sha256=cert['source_sha256'],
                    initialization='fresh public backbone pretrained; no prior trained checkpoint',
                    trained_A_B_previous_stage_checkpoint_loaded=False, evaluation_weights='ema',
                    selection_metric='AP50', output_directory=str(output), locks=paths,
                    stage='replacement_stage3', started_utc=datetime.now(timezone.utc).isoformat())
        write_launch_provenance(RUN, ROOT, cert['source_sha256'], cfg, info)
        with (RUN/'train.log').open('x') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     pass_fds=tuple(h.fileno() for h in locks.handles))
            (RUN/'pid.txt').write_text(str(child.pid)+'\n')
            publish(status='running', pid=child.pid, output=str(output), physical_mapping=mapping)
            while child.poll() is None:
                time.sleep(15)
                jobs = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True)
                actual = [line.split(',')[1].strip() for line in jobs.splitlines() if int(line.split(',')[0]) == child.pid]
                if actual and actual != [GPU_UUID]:
                    raise RuntimeError('Training process mapped to unexpected GPU')
                publish(physical_mapping_verified=bool(actual))
        (RUN/'exit_code.txt').write_text(str(child.returncode)+'\n')
        child = None
        finished = verify_completed_run(output, RUN, cfg)
        publish(status='completed', completion=finished)
        (QDIR/'exit_code.txt').write_text('0\n')
    except BaseException as error:
        publish(status='stopped' if isinstance(error, (SystemExit,KeyboardInterrupt)) else 'failed', error=str(error))
        (QDIR/'exit_code.txt').write_text(str(error.code if isinstance(error,SystemExit) else 1)+'\n')
        raise
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:child.kill();child.wait()
            (RUN/'exit_code.txt').write_text(str(child.returncode)+'\n')
        if locks is not None:locks.release()
        controller_lock.release()


if __name__ == '__main__':
    main()
