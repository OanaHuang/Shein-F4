"""Queue orchestration and execution guards; no real training or data."""

from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import f4.common as common
SPEC = importlib.util.spec_from_file_location('local_training_under_test', ROOT / 'scripts/train_local.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class FakeProcess:
    def __init__(self, run_id, events, code=0, running=False):
        self.run_id, self.events, self.returncode = run_id, events, code
        self.running = running
        self.pid = 12345

    def poll(self):
        return None if self.running else self.returncode

    def wait(self, timeout=None):
        self.events.append(('wait', self.run_id))
        self.running = False
        return self.returncode

    def terminate(self):
        self.events.append(('terminate', self.run_id))
        self.running = False
        self.returncode = -15


@contextmanager
def simulated_source_change():
    original, _ = common.source_snapshot()
    changed = [False]

    def snapshot():
        hashes = dict(original)
        if changed[0]:
            hashes['src/f4/models.py'] = '0' * 64
        return hashes, common.configuration_hash(hashes)

    with patch.object(common, 'source_snapshot', side_effect=snapshot):
        yield changed


class LocalTrainingQueueTests(unittest.TestCase):
    def prepare(self, root, jobs=2):
        config = {'experiment': 'B1', 'model': {'n_jobs': 4},
                  'execution': {'stage': 'training', 'allow_real_training': True,
                                'parallel_models': False, 'cpu_threads': 4}}
        (root / 'config.yaml').write_text(yaml.safe_dump(config))
        plan = {'batch_id': 'test_batch', 'max_rss_gib': 1, 'poll_seconds': 0.01,
                'jobs': [{'run_id': f'run_{i}', 'config': 'config.yaml', 'fold': 'dev_1'}
                         for i in range(jobs)]}
        path = root / 'plan.yaml'
        path.write_text(yaml.safe_dump(plan))
        return path, plan

    def status(self, root):
        return json.loads((root / 'reports/training/test_batch/status.json').read_text())

    def completed_run(self, root, run_id):
        folder = root / 'reports/runs' / run_id
        folder.mkdir(parents=True, exist_ok=True)
        manifest = {'run_id': run_id, 'status': 'complete', 'synthetic': False,
                    'metrics': {'A': 1.0}}
        (folder / 'run_manifest.json').write_text(json.dumps(manifest))
        (folder / 'resources.json').write_text(json.dumps({'run_id': run_id,
                    'status': 'complete', 'peak_RSS_bytes': 1024}))
        return manifest

    def assert_run_status(self, root, expected):
        folder = root / 'reports/runs/run_0'
        for name in ['run_manifest.json', 'resources.json']:
            self.assertEqual(json.loads((folder / name).read_text())['status'], expected, name)
        self.assertEqual(self.status(root)['jobs'][0]['status'], expected)

    def test_authorization_is_checked_before_reading_a_plan(self):
        with patch.object(Path, 'read_text') as read:
            with self.assertRaisesRegex(ValueError, 'execute-training'):
                runner.run_batch('does-not-exist.yaml')
            read.assert_not_called()

    def test_holdout_preparation_and_excess_threads_are_rejected(self):
        with TemporaryDirectory() as directory, patch.object(runner, 'ROOT', Path(directory)):
            root = Path(directory)
            _, plan = self.prepare(root)
            with self.assertRaisesRegex(ValueError, 'holdout'):
                runner.validate_job(dict(plan['jobs'][0], fold='holdout'))
            config = yaml.safe_load((root / 'config.yaml').read_text())
            config['execution']['allow_real_training'] = False
            (root / 'config.yaml').write_text(yaml.safe_dump(config))
            with self.assertRaisesRegex(ValueError, 'explicit'):
                runner.validate_job(plan['jobs'][0])
            config['execution']['allow_real_training'] = True
            config['model']['n_jobs'] = 8
            (root / 'config.yaml').write_text(yaml.safe_dump(config))
            with self.assertRaisesRegex(ValueError, 'four threads'):
                runner.validate_job(plan['jobs'][0])

    def test_jobs_run_serially_and_existing_batch_cannot_be_overwritten(self):
        with TemporaryDirectory() as directory, patch.object(runner, 'ROOT', Path(directory)):
            root = Path(directory)
            path, _ = self.prepare(root)
            events, commands = [], []

            def launch(command, **kwargs):
                commands.append(command)
                run_id = command[command.index('--run-id') + 1]
                events.append(('start', run_id))
                self.completed_run(root, run_id)
                return FakeProcess(run_id, events)

            with patch.object(runner.subprocess, 'Popen', side_effect=launch):
                result = runner.run_batch(path, execute_training=True)
            self.assertEqual(events, [('start', 'run_0'), ('wait', 'run_0'),
                                      ('start', 'run_1'), ('wait', 'run_1')])
            self.assertEqual(result['status'], 'complete')
            self.assertTrue(all(job['status'] == 'complete' for job in result['jobs']))
            self.assertTrue(all('--execution-snapshot' in command for command in commands))
            snapshot_path = Path(commands[0][commands[0].index('--execution-snapshot') + 1])
            snapshot = json.loads(snapshot_path.read_text())
            self.assertIn('official/tools/score.py', snapshot['files_sha256'])
            self.assertIn('environment', snapshot)
            with patch.object(runner.subprocess, 'Popen') as launch:
                with self.assertRaises(FileExistsError):
                    runner.run_batch(path, execute_training=True)
                launch.assert_not_called()

    def test_worker_failure_stops_before_the_next_job(self):
        with TemporaryDirectory() as directory, patch.object(runner, 'ROOT', Path(directory)):
            root = Path(directory)
            path, _ = self.prepare(root)
            with patch.object(runner.subprocess, 'Popen', return_value=FakeProcess('run_0', [], code=7)) as launch:
                with self.assertRaisesRegex(RuntimeError, 'status 7'):
                    runner.run_batch(path, execute_training=True)
                self.assertEqual(launch.call_count, 1)
            state = self.status(root)
            self.assertEqual(state['status'], 'failed')
            self.assertEqual([job['status'] for job in state['jobs']], ['failed', 'pending'])

    def test_rss_limit_terminates_worker_and_stops_queue(self):
        with TemporaryDirectory() as directory, patch.object(runner, 'ROOT', Path(directory)):
            root = Path(directory)
            path, _ = self.prepare(root)
            events = []
            process = FakeProcess('run_0', events, running=True)
            with patch.object(runner.subprocess, 'Popen', return_value=process) as launch, \
                    patch.object(runner, 'read_rss_bytes', return_value=2 * 1024 ** 3):
                with self.assertRaisesRegex(MemoryError, 'RSS exceeded'):
                    runner.run_batch(path, execute_training=True)
                self.assertEqual(launch.call_count, 1)
            self.assertIn(('terminate', 'run_0'), events)
            state = self.status(root)
            self.assertEqual(state['jobs'][0]['sampled_peak_RSS_bytes'], 2 * 1024 ** 3)
            self.assertEqual(state['status'], 'failed')
            self.assertIn('heartbeat_at', state)

    def test_worker_records_resources_without_reading_real_data(self):
        with TemporaryDirectory() as directory, patch.object(runner, 'ROOT', Path(directory)):
            root = Path(directory)
            _, plan = self.prepare(root, jobs=1)
            resources = root / 'reports/runs/run_0/resources.json'
            with patch('f4.backtest.run_backtest', side_effect=lambda *a, **k: self.completed_run(root, 'run_0')) as run, \
                    patch.object(runner.resource, 'getrusage', return_value=SimpleNamespace(ru_maxrss=4096)), \
                    patch.object(runner.sys, 'platform', 'darwin'):
                runner.worker(plan['jobs'][0], resources, execute_training=True)
            run.assert_called_once()
            self.assertTrue(run.call_args.kwargs['execute_training'])
            report = json.loads(resources.read_text())
            self.assertEqual(report['status'], 'complete')
            self.assertEqual(report['peak_RSS_bytes'], 4096)
            self.assertGreaterEqual(report['elapsed_seconds'], 0)

    def test_queue_sigterm_stops_worker_and_persists_failure(self):
        with TemporaryDirectory() as directory, patch.object(runner, 'ROOT', Path(directory)):
            root = Path(directory)
            path, _ = self.prepare(root)
            events = []
            process = FakeProcess('run_0', events, running=True)
            old_handler = signal.getsignal(signal.SIGTERM)
            with patch.object(runner.subprocess, 'Popen', return_value=process), \
                    patch.object(runner, 'read_rss_bytes', side_effect=lambda pid: signal.raise_signal(signal.SIGTERM)):
                with self.assertRaisesRegex(KeyboardInterrupt, 'termination request'):
                    runner.run_batch(path, execute_training=True)
            self.assertIn(('terminate', 'run_0'), events)
            self.assertEqual(self.status(root)['status'], 'failed')
            self.assertEqual(signal.getsignal(signal.SIGTERM), old_handler)

    def test_last_job_source_change_invalidates_all_completion_records(self):
        with TemporaryDirectory() as directory, patch.object(runner, 'ROOT', Path(directory)), \
                simulated_source_change() as changed:
            root = Path(directory)
            path, _ = self.prepare(root, jobs=1)

            def launch(command, **kwargs):
                self.completed_run(root, 'run_0')
                changed[0] = True
                return FakeProcess('run_0', [])

            with patch.object(runner.subprocess, 'Popen', side_effect=launch):
                with self.assertRaises(common.ExecutionInvalidatedError):
                    runner.run_batch(path, execute_training=True)
            self.assertEqual(self.status(root)['status'], 'invalidated')
            self.assert_run_status(root, 'invalidated')

    def test_last_job_configuration_change_invalidates_all_completion_records(self):
        with TemporaryDirectory() as directory, patch.object(runner, 'ROOT', Path(directory)):
            root = Path(directory)
            path, _ = self.prepare(root, jobs=1)

            def launch(command, **kwargs):
                self.completed_run(root, 'run_0')
                config = root / 'config.yaml'
                config.write_text(config.read_text() + '\n# changed during final job\n')
                return FakeProcess('run_0', [])

            with patch.object(runner.subprocess, 'Popen', side_effect=launch):
                with self.assertRaises(common.ExecutionInvalidatedError):
                    runner.run_batch(path, execute_training=True)
            self.assertEqual(self.status(root)['status'], 'invalidated')
            self.assert_run_status(root, 'invalidated')

    def test_worker_source_change_invalidates_manifest_and_resources(self):
        with TemporaryDirectory() as directory, patch.object(runner, 'ROOT', Path(directory)), \
                simulated_source_change() as changed:
            root = Path(directory)
            _, plan = self.prepare(root, jobs=1)

            def complete_then_drift(*args, **kwargs):
                result = self.completed_run(root, 'run_0')
                changed[0] = True
                return result

            resources = root / 'reports/runs/run_0/resources.json'
            with patch('f4.backtest.run_backtest', side_effect=complete_then_drift):
                with self.assertRaises(common.ExecutionInvalidatedError):
                    runner.worker(plan['jobs'][0], resources, execute_training=True)
            folder = root / 'reports/runs/run_0'
            self.assertEqual(json.loads((folder / 'run_manifest.json').read_text())['status'], 'invalidated')
            self.assertEqual(json.loads(resources.read_text())['status'], 'invalidated')

    def test_three_unavailable_rss_samples_stop_before_next_job(self):
        with TemporaryDirectory() as directory, patch.object(runner, 'ROOT', Path(directory)):
            root = Path(directory)
            path, _ = self.prepare(root)
            events = []
            process = FakeProcess('run_0', events, running=True)
            with patch.object(runner.subprocess, 'Popen', return_value=process) as launch, \
                    patch.object(runner, 'read_rss_bytes', return_value=None) as sample, \
                    patch.object(runner.time, 'sleep'):
                with self.assertRaises(runner.ResourceMonitorError):
                    runner.run_batch(path, execute_training=True)
            self.assertEqual(sample.call_count, 3)
            self.assertEqual(launch.call_count, 1)
            self.assertIn(('terminate', 'run_0'), events)
            state = self.status(root)
            self.assertEqual(state['status'], 'failed')
            self.assertEqual(state['failure_kind'], 'resource_monitor_failed')
            self.assertEqual(state['jobs'][1]['status'], 'pending')

    def test_temporary_rss_failures_recover(self):
        with TemporaryDirectory() as directory, patch.object(runner, 'ROOT', Path(directory)):
            root = Path(directory)
            path, _ = self.prepare(root, jobs=1)
            process = FakeProcess('run_0', [], running=True)
            samples = iter([None, None, 1024, None, 2048])
            sample_count = [0]

            def launch(*args, **kwargs):
                self.completed_run(root, 'run_0')
                return process

            def read(pid):
                sample_count[0] += 1
                value = next(samples)
                if sample_count[0] == 5:
                    process.running = False
                return value

            with patch.object(runner.subprocess, 'Popen', side_effect=launch), \
                    patch.object(runner, 'read_rss_bytes', side_effect=read), \
                    patch.object(runner.time, 'sleep'):
                result = runner.run_batch(path, execute_training=True)
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(sample_count[0], 5)
            self.assertGreaterEqual(result['jobs'][0]['sampled_peak_RSS_bytes'], 1024)
            self.assertNotIn('monitor_warning', result['jobs'][0])

    def test_exit_race_does_not_treat_final_missing_rss_as_failure(self):
        with TemporaryDirectory() as directory, patch.object(runner, 'ROOT', Path(directory)):
            root = Path(directory)
            path, _ = self.prepare(root, jobs=1)
            process = FakeProcess('run_0', [], running=True)

            def launch(*args, **kwargs):
                self.completed_run(root, 'run_0')
                return process

            def exited(pid):
                process.running = False
                return None

            with patch.object(runner.subprocess, 'Popen', side_effect=launch), \
                    patch.object(runner, 'read_rss_bytes', side_effect=exited):
                result = runner.run_batch(path, execute_training=True)
            self.assertEqual(result['status'], 'complete')

    def test_rss_permission_error_is_an_unavailable_sample(self):
        with patch.object(runner.subprocess, 'run', side_effect=PermissionError('denied')):
            self.assertIsNone(runner.read_rss_bytes(12345))

    def test_cli_worker_rejects_before_importing_core(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'scripts').mkdir()
            script = root / 'scripts/train_local.py'
            script.write_bytes((ROOT / 'scripts/train_local.py').read_bytes())
            package = root / 'src/f4'
            package.mkdir(parents=True)
            (package / '__init__.py').write_text('raise RuntimeError("F4_IMPORTED_BEFORE_GUARD")\n')
            snapshot = root / 'execution.json'
            snapshot.write_text('THIS_IS_NOT_JSON')
            command = [sys.executable, str(script), '--worker', '--execution-snapshot', str(snapshot),
                       '--config', 'config.yaml', '--fold', 'dev_1', '--run-id', 'run_0',
                       '--resource-report', 'reports/runs/run_0/resources.json']
            result = subprocess.run(command, cwd=root, text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('execute-training', result.stderr)
            self.assertNotIn('F4_IMPORTED_BEFORE_GUARD', result.stderr)
            snapshot.write_text(json.dumps({'schema_version': 1, 'files_sha256': {},
                'extra_files_sha256': {}, 'environment': {}, 'snapshot_sha256': 'invalid'}))
            result = subprocess.run(command + ['--execute-training'], cwd=root, text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn('F4_IMPORTED_BEFORE_GUARD', result.stderr)


if __name__ == '__main__':
    unittest.main()
