"""Small standard-library portability helpers for preparation tools."""

import hashlib
import os
from datetime import datetime, timezone


def utc_timestamp():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def file_digest(stream, algorithm='sha256'):
    digest = hashlib.new(algorithm)
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        digest.update(block)
    return digest


def current_process_peak_memory():
    """Read the actual Python worker's OS lifetime peaks, not a venv launcher."""
    if os.name != 'nt':
        return {'status': 'unavailable', 'reason': 'Windows counters only'}
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in ('PeakWorkingSetSize', 'WorkingSetSize',
            'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
            'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    query = ctypes.WinDLL('psapi', use_last_error=True).GetProcessMemoryInfo
    query.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    query.restype = wintypes.BOOL
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    if not query(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        return {'status': 'unavailable', 'windows_error': ctypes.get_last_error()}
    return {'status': 'available', 'pid': os.getpid(),
            'scope': 'Actual Python worker OS lifetime peaks through completed backtest; excludes other processes',
            'peak_working_set_bytes': counters.PeakWorkingSetSize,
            'peak_commit_bytes': counters.PeakPagefileUsage}


def lock_file(stream):
    """Nonblocking exclusive lock, released when the stream closes."""
    if os.name == 'nt':
        import msvcrt
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write('0')
            stream.flush()
        stream.seek(0)
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise BlockingIOError('Another process holds the file lock') from exc
    else:
        import fcntl
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
