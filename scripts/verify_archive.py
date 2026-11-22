"""Check the official ZIP and optionally extract it, without loading CSVs."""

import argparse
import json
import shutil
import stat
import sys
from pathlib import Path
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from f4.common import atomic_write_json
from f4.platform_utils import file_digest, utc_timestamp, lock_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=Path('data/raw/F4.zip'))
    parser.add_argument('--extract', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    archive = args.archive if args.archive.is_absolute() else root / args.archive
    manifest_path = root / 'reports/data_manifest.json'
    manifest = (json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists()
                else {'archive': {'expected_bytes': 590772903, 'extracted': False},
                      'training_status': 'not_started'})
    expected = manifest['archive'].get('expected_bytes')
    if expected is not None and archive.stat().st_size != expected:
        raise ValueError(f'Incomplete download: {archive.stat().st_size} != {expected}')
    target = root / 'data/raw/official'
    with ZipFile(archive) as zipped:
        entries = zipped.infolist()
        for item in entries:
            if not (target / item.filename).resolve().is_relative_to(target.resolve()):
                raise ValueError(f'Unsafe ZIP path: {item.filename}')
            if stat.S_ISLNK(item.external_attr >> 16):
                raise ValueError(f'Symlink in ZIP: {item.filename}')
        names = {Path(item.filename).name for item in entries}
        required = {'train_202608_0920.csv', 'test_202609.csv', 'score.py'}
        if required - names:
            raise ValueError(f'Missing expected files: {required - names}')
        bad = zipped.testzip()
        if bad:
            raise ValueError(f'ZIP CRC failed: {bad}')
        uncompressed = sum(item.file_size for item in entries)
        inventory = [dict(path=item.filename, bytes=item.file_size,
                          compressed_bytes=item.compress_size, crc32=f'{item.CRC:08x}')
                     for item in entries if not item.is_dir()]
        with archive.open('rb') as stream:
            digest = file_digest(stream, 'sha256').hexdigest()
        previous_digest = manifest['archive'].get('sha256')
        if args.extract:
            if target.exists() and any(target.iterdir()):
                if not (manifest['archive'].get('extracted') and previous_digest == digest):
                    raise ValueError('Extraction directory is nonempty; inspect before replacing originals')
            else:
                if shutil.disk_usage(root).free < uncompressed + 15 * 1024**3:
                    raise OSError('Insufficient space for extraction plus 15GiB system reserve')
                zipped.extractall(target)
            manifest['archive']['extracted'] = True
            # ZIP and Git may encode line endings differently; originals stay intact.
            scorer = next(target / item.filename for item in entries
                          if Path(item.filename).name == 'score.py')
            tracked = root / 'official/tools/score.py'
            manifest['archive']['scorer_text_matches'] = (
                scorer.read_text(encoding='utf-8') == tracked.read_text(encoding='utf-8'))
            if not manifest['archive']['scorer_text_matches']:
                raise ValueError('Archive scorer differs from tracked scorer; originals preserved')
        manifest['archive'].update(bytes=archive.stat().st_size, sha256=digest,
                                   zip_verified=True, uncompressed_bytes=uncompressed,
                                   inventory=inventory, status='verified',
                                   downloaded_bytes=archive.stat().st_size)
    if archive.suffix == '.part':
        final_path = archive.with_suffix('')
        if final_path.exists():
            raise FileExistsError(final_path)
        archive.rename(final_path)
        archive = final_path
    manifest['updated_at'] = utc_timestamp()
    manifest['archive']['target'] = str(archive)
    atomic_write_json(manifest_path, manifest)
    print(json.dumps({'archive': str(archive), 'bytes': archive.stat().st_size,
                      'sha256': digest, 'uncompressed_bytes': uncompressed,
                      'extracted': manifest['archive']['extracted']}, ensure_ascii=False))


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    (root / 'reports/downloads').mkdir(parents=True, exist_ok=True)
    with (root / 'reports/downloads/archive_verify.lock').open('a') as lock:
        lock_file(lock)
        main()
