import os
from pathlib import Path
import subprocess
import unittest


class TestICBFCLauncher(unittest.TestCase):

    def test_dry_run_has_three_fresh_ordered_gpu0_commands(self):
        environment = os.environ.copy()
        environment['ICBFC_DRY_RUN'] = '1'

        completed = subprocess.run(
            ['bash', 'tools/run_icbfc_seeds_gpu0.sh'],
            check=True, capture_output=True, text=True, env=environment)

        lines = [line for line in completed.stdout.splitlines()
                 if line.startswith('RUN seed=')]
        self.assertEqual(len(lines), 3)
        self.assertEqual(
            [line.split()[1] for line in lines],
            ['seed=0', 'seed=1', 'seed=2'])
        for seed, line in enumerate(lines):
            self.assertIn(
                'configs/coxnet/icbfc/ICBFC.py', line)
            self.assertIn(f'icbfc/seed{seed}', line)
            self.assertIn('--gpu-id 0', line)
            self.assertIn('--deterministic', line)
            self.assertNotIn('--auto-resume', line)
        self.assertIn('CUDA_VISIBLE_DEVICES=0', completed.stdout)

    def test_launcher_has_lock_and_fresh_directory_guard(self):
        source = Path('tools/run_icbfc_seeds_gpu0.sh').read_text()

        self.assertIn('/tmp/coxnet_icbfc_gpu0.lock', source)
        self.assertIn('flock', source)
        self.assertIn('ICBFC_ALLOW_EXISTING', source)
        self.assertNotIn('--auto-resume', source)


if __name__ == '__main__':
    unittest.main()
