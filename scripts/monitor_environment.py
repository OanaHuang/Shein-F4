#!/usr/bin/env python3
"""Watch an already-running project pip process, then verify its environment."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from f4.platform_utils import utc_timestamp

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/environment_check.json"
PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "pyarrow": "pyarrow",
    "lightgbm": "lightgbm",
    "scikit-learn": "sklearn",
    "pyyaml": "yaml",
}


def timestamp() -> str:
    return utc_timestamp()


def save(report: dict) -> None:
    report["checked_at"] = timestamp()
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    pending = REPORT.with_suffix(".json.tmp")
    pending.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(REPORT)


def is_running(pid: int) -> bool:
    if os.name == 'nt':
        import ctypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.restype = ctypes.c_void_p
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return ctypes.get_last_error() == 5
        code = ctypes.c_ulong()
        try:
            if not kernel.GetExitCodeProcess(ctypes.c_void_p(handle), ctypes.byref(code)):
                raise OSError(ctypes.get_last_error(), 'GetExitCodeProcess failed')
            return code.value == 259
        finally:
            kernel.CloseHandle(ctypes.c_void_p(handle))
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def wheel_progress() -> list[dict]:
    result = []
    for directory in Path(tempfile.gettempdir()).glob("pip-unpack-*"):
        for wheel in directory.glob("*.whl"):
            try:
                result.append({"file": wheel.name, "downloaded_bytes": wheel.stat().st_size})
            except FileNotFoundError:
                pass
    return sorted(result, key=lambda item: item["file"])


def run_check(code: str, timeout: int = 60) -> dict:
    try:
        process = subprocess.run(
            [sys.executable, "-c", code], cwd=ROOT,
            text=True, capture_output=True, timeout=timeout,
        )
        return {
            "passed": process.returncode == 0,
            "returncode": process.returncode,
            "stdout": process.stdout[-6000:],
            "stderr": process.stderr[-6000:],
        }
    except subprocess.TimeoutExpired:
        return {"passed": False, "error": f"check timed out after {timeout}s"}


def verify(report: dict) -> None:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    report["packages"] = versions
    report["pip_process_running"] = False
    report["pip_exit_code"] = None
    report["pip_exit_code_note"] = "The observed pip process is not a child of this monitor."
    # Environment checks must never rewrite the repository dependency lock.
    report['python'] = sys.version
    report['python_executable'] = sys.executable
    report['platform'] = platform.platform()
    report['requirements_lock_written'] = False

    checks = {}
    for package, module in PACKAGES.items():
        checks[package] = run_check(
            f"import importlib; m=importlib.import_module({module!r}); "
            "print(getattr(m, '__version__', 'imported'))"
        )
    report["import_checks"] = checks
    report["imports_verified"] = all(check["passed"] for check in checks.values())
    report["lightgbm_training_check"] = run_check("""
import json
import numpy as np
from lightgbm import LGBMRegressor
rng = np.random.default_rng(2026)
x = rng.normal(size=(100, 4)).astype(np.float32)
y = 10 + 2 * x[:, 0] + rng.random(100)
model = LGBMRegressor(objective='quantile', alpha=0.9, n_estimators=5,
                     num_leaves=7, min_child_samples=5, n_jobs=1,
                     force_col_wise=True, verbosity=-1, random_state=2026)
model.fit(x, y)
pred = model.predict(x[:5])
assert pred.shape == (5,) and np.isfinite(pred).all()
print(json.dumps({'rows': 100, 'features': 4, 'trees': 5,
                  'prediction_min': float(pred.min()),
                  'prediction_max': float(pred.max()),
                  'purpose': 'synthetic environment check; no F4 data used'}))
""")
    report["lightgbm_training_verified"] = report["lightgbm_training_check"]["passed"]
    dependency_check = subprocess.run(
        [sys.executable, "-m", "pip", "check"], cwd=ROOT,
        text=True, capture_output=True, timeout=60,
    )
    report["dependency_check"] = {
        "passed": dependency_check.returncode == 0,
        "stdout": dependency_check.stdout,
        "stderr": dependency_check.stderr,
    }
    if not all(versions.values()):
        report["status"] = "installation_incomplete"
    elif report["imports_verified"] and report["lightgbm_training_verified"] and report["dependency_check"]["passed"]:
        report["status"] = "ready"
    else:
        report["status"] = "installed_with_runtime_errors"
    save(report)
    print(f"{timestamp()} verification complete: {report['status']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pip-pid", required=True, type=int)
    arguments = parser.parse_args()
    report = {
        "started_at": timestamp(),
        "status": "installing",
        "python": sys.version,
        "python_executable": sys.executable,
        "architecture": platform.machine(),
        "venv": str(ROOT / ".venv"),
        "installation_scope": "project virtual environment only",
        "requirements_source": "requirements.txt",
        "pip_pid": arguments.pip_pid,
        "monitor_pid": os.getpid(),
        "monitor_log": "reports/environment_install.log",
        "default_network_attempt": "failed: DNS unavailable in sandbox",
        "network_authorized_attempt": "existing pip process retained without restarting downloads",
        "imports_verified": False,
        "lightgbm_training_verified": False,
        "requirements_lock_written": False,
    }
    try:
        while is_running(arguments.pip_pid):
            report["pip_process_running"] = True
            report["wheel_progress"] = wheel_progress()
            save(report)
            print(f"{timestamp()} pip pid={arguments.pip_pid}; wheels={report['wheel_progress']}", flush=True)
            time.sleep(30)
        verify(report)
    except Exception as exc:
        report["status"] = "monitor_error"
        report["error"] = f"{type(exc).__name__}: {exc}"
        save(report)
        raise


if __name__ == "__main__":
    main()
