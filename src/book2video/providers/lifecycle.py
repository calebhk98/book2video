from __future__ import annotations

import asyncio
import os
import shlex
from typing import Mapping


async def run_lifecycle_command(
    command: list[str] | str | None,
    *,
    timeout_sec: float = 300.0,
    env: Mapping[str, str] | None = None,
) -> None:
    """Run an optional provider lifecycle command.

    To support local runtimes that are not controlled through a common Python API, profiles can
    use small shell commands to start/preload or stop/unload a server around a phase boundary.
    """
    if not command:
        return
    argv = shlex.split(command) if isinstance(command, str) else list(command)
    proc_env = os.environ.copy()
    if env:
        proc_env.update(env)
    proc = await asyncio.create_subprocess_exec(*argv, env=proc_env)
    try:
        code = await asyncio.wait_for(proc.wait(), timeout=timeout_sec)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        raise RuntimeError(f"Lifecycle command timed out: {argv!r}")
    if code != 0:
        raise RuntimeError(f"Lifecycle command exited with code {code}: {argv!r}")
