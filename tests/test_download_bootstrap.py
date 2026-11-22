"""Download initialization checks; no networking or real F4 files."""

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('download_bootstrap_under_test',
                                            ROOT / 'scripts/fetch_official.py')
downloader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(downloader)


class DownloadBootstrapTests(unittest.TestCase):
    def workspace(self, root):
        return patch.multiple(downloader, ROOT=root,
                              MANIFEST=root / 'reports/data_manifest.json',
                              PARTIAL=root / 'data/raw/F4.zip.part')

    def initialize_under_lock(self, root):
        downloader.prepare_download_directories()
        with (root / 'reports/downloads/F4.lock').open('a') as lock:
            downloader.lock_file(lock)
            downloader.initialize_manifest()

    def test_fresh_workspace_has_honest_initial_download_state(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with self.workspace(root):
                self.initialize_under_lock(root)
                manifest = json.loads(downloader.MANIFEST.read_text(encoding='utf-8'))
            self.assertTrue((root / 'reports/downloads').is_dir())
            self.assertTrue((root / 'data/raw').is_dir())
            archive = manifest['archive']
            self.assertEqual(archive['source_url'],
                             'https://imupd-public.sheincorp.cn/d/external-contents/F4.zip')
            self.assertEqual(archive['expected_bytes'], 590772903)
            self.assertEqual(archive['target'], 'data/raw/F4.zip')
            self.assertEqual(archive['partial_target'], 'data/raw/F4.zip.part')
            self.assertEqual(archive['status'], 'not_started')
            self.assertEqual(archive['downloaded_bytes'], 0)
            self.assertEqual(archive['bytes'], 0)
            self.assertFalse(archive['zip_verified'])
            self.assertFalse(archive['extracted'])
            self.assertIsNone(archive['sha256'])
            self.assertFalse((root / archive['target']).exists())
            self.assertFalse((root / archive['partial_target']).exists())

    def test_existing_manifest_bytes_are_preserved(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / 'reports/data_manifest.json'
            manifest.parent.mkdir()
            original = b'{"archive": {"status": "waiting_for_network", "downloaded_bytes": 123}, "custom": true}\n'
            manifest.write_bytes(original)
            with self.workspace(root):
                self.initialize_under_lock(root)
                self.initialize_under_lock(root)
            self.assertEqual(manifest.read_bytes(), original)

    def test_main_does_not_initialize_when_download_lock_is_held(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with self.workspace(root), patch.object(downloader.os, 'chdir'), \
                    patch.object(downloader, 'lock_file', side_effect=BlockingIOError), \
                    patch.object(downloader.subprocess, 'Popen') as download, \
                    patch.object(downloader.subprocess, 'run') as verify:
                with self.assertRaisesRegex(SystemExit, 'already running'):
                    downloader.main()
                self.assertFalse(downloader.MANIFEST.exists())
                download.assert_not_called()
                verify.assert_not_called()


if __name__ == '__main__':
    unittest.main()
