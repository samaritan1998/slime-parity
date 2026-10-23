"""Optional local subprocess runner; external schedulers can seal their own runs."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import math

from .common import require, inside, write_new
from .plan import load_plan, verify_artifacts
from .storage import finalize_stage


def stop(process):
    if process is None:
        return
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        # The parent may exit before its children; finish the owned process group.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    elif process.poll() is None:
        process.kill()
    process.wait()


def run_stage(plan_path, name, *, timeout=3600):
    """Launch a frozen argv and seal evidence after successful process exit.

    POSIX timeouts terminate the owned process group. On Windows only the direct
    child is terminated; use a scheduler for process-tree lifecycle management.
    """
    path = Path(plan_path).resolve()
    plan = load_plan(path)
    require(name in plan["stages"], "Unknown stage")
    require(type(timeout) in (int, float) and math.isfinite(timeout) and timeout > 0, "Invalid timeout")
    stage = plan["stages"][name]
    require("command" in stage, "Stage has no command; use an external launcher")
    verify_artifacts(plan)
    folder = inside(path.parent, name)
    folder.mkdir(exist_ok=False)  # Exclusive claim, including after a failed run.
    replacements = {"python": sys.executable, "plan": str(path), "stage": name,
                    "world_size": str(stage["world_size"]), "output": str(folder)}
    command = [arg.format_map(replacements) for arg in stage["command"]]
    env = dict(os.environ)
    env.update(stage.get("environment", {}))
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
    process, started = None, time.monotonic()
    try:
        with (folder / "train.log").open("x", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=folder, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=(os.name == "posix"))
            returncode = process.wait(timeout=timeout)
        require(returncode == 0, f"Training process exited {returncode}; see train.log")
        return finalize_stage(path, name, returncode=returncode, provenance="local_runner", command=command)
    except BaseException as error:
        stop(process)
        write_new(folder / "failure.json", {"status": "execution_failure",
                  "error_type": type(error).__name__, "error": str(error),
                  "returncode": None if process is None else process.returncode,
                  "seconds": time.monotonic() - started})
        raise
