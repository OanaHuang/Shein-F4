"""Project paths, deterministic hashes and small atomic metadata files."""

from datetime import datetime
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def timestamp():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def sha256_file(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, 'item'):
        return _plain(value.item())
    if hasattr(value, 'isoformat'):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(_plain(obj), ensure_ascii=False, indent=2,
                                    sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def configuration_hash(value):
    raw = json.dumps(_plain(value), sort_keys=True, ensure_ascii=False,
                     separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def source_snapshot():
    files = []
    for directory, pattern in [('src', '*.py'), ('scripts', '*.py'), ('tests', '*.py'), ('configs', '*.yaml')]:
        files.extend((PROJECT_ROOT / directory).rglob(pattern))
    files.extend(PROJECT_ROOT / name for name in ['AGENTS.md', 'README.md', 'pyproject.toml',
                                                   'requirements.txt', 'requirements.lock.txt'])
    hashes = {str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in sorted(files)}
    return hashes, configuration_hash(hashes)


class ExecutionInvalidatedError(RuntimeError):
    """The files/configuration/environment no longer match the execution start."""

    def __init__(self, message, observed=None):
        super().__init__(message)
        self.observed = observed


def execution_environment():
    versions = {}
    for name in ['numpy', 'pandas', 'pyarrow', 'lightgbm', 'scikit-learn', 'PyYAML', 'xgboost', 'ngboost']:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {'python_version': sys.version, 'python_executable': str(Path(sys.executable).absolute()),
            'python_prefix': str(Path(sys.prefix).absolute()), 'platform': platform.system(),
            'machine': platform.machine(), 'packages': versions}


def execution_snapshot(extra_paths=()):
    """Capture source, official scoring, selected config files and runtime identity.

    Start/end checks detect persistent drift, not edits reverted between checks.
    Use an immutable source export when stronger execution isolation is needed.
    """
    files, _ = source_snapshot()
    files['official/tools/score.py'] = sha256_file(PROJECT_ROOT / 'official/tools/score.py')
    payload = {'schema_version': 1, 'files_sha256': files,
               'extra_files_sha256': {str(Path(p).resolve()): sha256_file(p)
                                      for p in sorted({str(Path(p).resolve()) for p in extra_paths})},
               'environment': execution_environment()}
    return {**payload, 'snapshot_sha256': configuration_hash(payload)}


def verify_execution_snapshot(expected, observed=None):
    """Raise on drift rather than treating end-of-run disk state as executed code."""
    if not isinstance(expected, dict) or expected.get('schema_version') != 1:
        raise ExecutionInvalidatedError('Invalid execution snapshot')
    payload = {key: value for key, value in expected.items() if key != 'snapshot_sha256'}
    if configuration_hash(payload) != expected.get('snapshot_sha256'):
        raise ExecutionInvalidatedError('Execution snapshot digest is invalid')
    if observed is None:
        try:
            observed = execution_snapshot(extra_paths=expected.get('extra_files_sha256', {}))
        except (OSError, ValueError) as error:
            raise ExecutionInvalidatedError(f'Execution snapshot cannot be verified: {error}') from error
    if observed.get('snapshot_sha256') != expected['snapshot_sha256']:
        raise ExecutionInvalidatedError('Execution source/configuration/environment changed', observed)
    return observed
