"""Queue orchestration checks with mocked workers; no real training or data."""

import importlib.util
import json
from pathlib import Path
import signal
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[1]
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
        with TemporaryDirectory() as directory, patch.object(runner, 'ROOT', Path(directory)), \
                patch.object(runner, 'source_snapshot', return_value=({}, 'fixed')):
            root = Path(directory)
            path, _ = self.prepare(root)
            events = []

            def launch(command, **kwargs):
                run_id = command[command.index('--run-id') + 1]
                events.append(('start', run_id))
                report = root / 'reports/runs' / run_id / 'run_manifest.json'
                report.parent.mkdir(parents=True)
                report.write_text(json.dumps({'status': 'complete', 'synthetic': False,
                                              'metrics': {'A': 1.0}}))
                return FakeProcess(run_id, events)

            with patch.object(runner.subprocess, 'Popen', side_effect=launch):
                result = runner.run_batch(path, execute_training=True)
            self.assertEqual(events, [('start', 'run_0'), ('wait', 'run_0'),
                                      ('start', 'run_1'), ('wait', 'run_1')])
            self.assertEqual(result['status'], 'complete')
            self.assertTrue(all(job['status'] == 'complete' for job in result['jobs']))
            with patch.object(runner.subprocess, 'Popen') as launch:
                with self.assertRaises(FileExistsError):
                    runner.run_batch(path, execute_training=True)
                launch.assert_not_called()

    def test_worker_failure_stops_before_the_next_job(self):
        with TemporaryDirectory() as directory, patch.object(runner, 'ROOT', Path(directory)), \
                patch.object(runner, 'source_snapshot', return_value=({}, 'fixed')):
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
        with TemporaryDirectory() as directory, patch.object(runner, 'ROOT', Path(directory)), \
                patch.object(runner, 'source_snapshot', return_value=({}, 'fixed')):
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
            with patch('f4.backtest.run_backtest', return_value={'metrics': {'A': 1.0}}) as run, \
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
        with TemporaryDirectory() as directory, patch.object(runner, 'ROOT', Path(directory)), \
                patch.object(runner, 'source_snapshot', return_value=({}, 'fixed')):
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


if __name__ == '__main__':
    unittest.main()
