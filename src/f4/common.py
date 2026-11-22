"""Project paths, deterministic hashes and small atomic metadata files."""

import hashlib
from .platform_utils import file_digest, utc_timestamp
import json
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def timestamp():
    return utc_timestamp()


def sha256_file(path):
    with Path(path).open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


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
                                                   'requirements.txt', 'requirements.lock.txt', 'requirements.windows-py310.lock.txt'])
    hashes = {str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in sorted(files)}
    return hashes, configuration_hash(hashes)
