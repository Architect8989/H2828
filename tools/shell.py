import subprocess
import time
import hashlib
from core.delta import Delta

WHITELIST = {
    ("uname", "-a"),
}

def _snapshot():
    return {
        "t": time.time()
    }

def _run(cmd):
    if tuple(cmd) not in WHITELIST:
        raise PermissionError("Command not allowed")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=3,
        env={"PATH": "/bin:/usr/bin", "LANG": "C"},
    )

    return {
        "returncode": result.returncode,
        "stdout_hash": hashlib.sha256((result.stdout or "").encode()).hexdigest(),
        "stderr_hash": hashlib.sha256((result.stderr or "").encode()).hexdigest(),
    }

def probe_system_identity():
    before = _snapshot()
    obs = _run(["uname", "-a"])
    after = _snapshot()

    if obs["returncode"] != 0:
        raise subprocess.CalledProcessError(obs["returncode"], "uname -a")

    return Delta({
        "probe.type": "system_identity",
        "before.t": before["t"],
        "after.t": after["t"],
        "obs.returncode": obs["returncode"],
        "obs.stdout_hash": obs["stdout_hash"],
        "obs.stderr_hash": obs["stderr_hash"],
    })
