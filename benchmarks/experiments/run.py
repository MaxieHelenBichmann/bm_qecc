"""Run one function call with timeout and memory supervision.

This module deliberately knows nothing about benchmark suites, algorithms,
input generators, command-line arguments, or CSV files. Those concerns live
one layer above it.
"""

from __future__ import annotations

import importlib
import multiprocessing as mp
import os
import signal
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from multiprocessing.process import BaseProcess
from queue import Empty
from time import perf_counter
from types import ModuleType
from typing import Any


MEMORY_POLL_INTERVAL_SECONDS = 0.2

resource: ModuleType | None
try:
    resource = importlib.import_module("resource")
except ImportError:  # pragma: no cover - resource is Unix-only
    resource = None


@dataclass(frozen=True)
class RunResult:
    """Outcome of one supervised function call.

    ``result_is_expected`` is false for every failed execution. The three
    failure fields are intentionally separate: a timeout, a memory-limit hit,
    and another execution error cannot be confused by callers.

    For a completed supervised call, ``runtime`` is measured inside the worker
    around ``function(*inputs)``. Process startup, queue transfer, and parent
    supervision are excluded. A timeout has no completed worker measurement,
    so its runtime is the parent-observed capped wall time instead.
    """

    runtime: float
    result: Any
    expected: Any
    result_is_expected: bool
    timed_out: bool
    memory_exceeded: bool
    error: str | None

    @property
    def successful(self) -> bool:
        """Whether the call completed and returned the expected result."""
        return (
            self.result_is_expected
            and not self.timed_out
            and not self.memory_exceeded
            and self.error is None
        )


def _set_memory_limit(max_memory_bytes: int) -> None:
    """Limit address-space allocations in the worker process when supported."""
    if resource is None or not hasattr(resource, "RLIMIT_AS"):
        return

    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    new_hard = (
        max_memory_bytes
        if hard == resource.RLIM_INFINITY
        else min(hard, max_memory_bytes)
    )
    new_soft = (
        max_memory_bytes
        if soft == resource.RLIM_INFINITY
        else min(soft, max_memory_bytes)
    )
    if new_hard != resource.RLIM_INFINITY:
        new_soft = min(new_soft, new_hard)

    try:
        resource.setrlimit(resource.RLIMIT_AS, (new_soft, new_hard))
    except (OSError, ValueError):
        # Parent-side RSS supervision remains active if RLIMIT_AS cannot be set.
        pass


def _worker(
    function: Callable[..., Any],
    inputs: tuple[Any, ...],
    queue: mp.Queue,
    max_memory_bytes: int | None,
) -> None:
    """Execute and time only the requested call in a supervised worker."""
    if hasattr(os, "setsid"):
        os.setsid()
    if max_memory_bytes is not None:
        _set_memory_limit(max_memory_bytes)

    start = perf_counter()
    try:
        result = function(*inputs)
        queue.put(("result", result, perf_counter() - start))
    except MemoryError:
        queue.put(("memory", None, perf_counter() - start))
    except BaseException as exc:  # noqa: BLE001 - the supervisor reports all failures
        queue.put(
            ("error", f"{type(exc).__name__}: {exc}", perf_counter() - start)
        )


def _process_group_rss_bytes(process_group_id: int) -> int | None:
    """Return combined resident memory of a worker process group, if observable."""
    try:
        completed = subprocess.run(
            ["ps", "-o", "rss=", "-g", str(process_group_id)],
            check=False,
            capture_output=True,
            text=True,
            timeout=1,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None

    values: list[int] = []
    for line in completed.stdout.splitlines():
        try:
            values.append(int(line.strip()))
        except ValueError:
            continue
    return sum(values) * 1024 if values else None


def _terminate_process_group(process: BaseProcess) -> None:
    """Terminate the worker and subprocesses it created."""
    if process.pid is None:
        return

    try:
        if hasattr(os, "killpg"):
            os.killpg(process.pid, signal.SIGTERM)
        else:  # pragma: no cover - exercised on Windows
            process.terminate()
    except ProcessLookupError:
        return
    except OSError:
        process.terminate()

    process.join(5)
    if not process.is_alive():
        return

    try:
        if hasattr(os, "killpg"):
            os.killpg(process.pid, signal.SIGKILL)
        else:  # pragma: no cover - exercised on Windows
            process.kill()
    except ProcessLookupError:
        return
    except OSError:
        process.kill()
    process.join()


def _matches_expected(result: Any, expected: Any) -> bool:
    """Compare a returned value with the caller-provided expected value."""
    try:
        return bool(result == expected)
    except (TypeError, ValueError):
        return False


def _failed_result(
    *,
    runtime: float,
    expected: Any,
    timed_out: bool = False,
    memory_exceeded: bool = False,
    error: str | None = None,
) -> RunResult:
    return RunResult(
        runtime=runtime,
        result=None,
        expected=expected,
        result_is_expected=False,
        timed_out=timed_out,
        memory_exceeded=memory_exceeded,
        error=error,
    )


def run(
    function: Callable[..., Any],
    inputs: Sequence[Any],
    expected: Any,
    timeout: float | None = None,
    max_memory_bytes: int | None = None,
) -> RunResult:
    """Run exactly one function call and report its supervised outcome.

    Args:
        function: Function to execute.
        inputs: Positional arguments passed to ``function``.
        expected: Exact expected return value.
        timeout: Optional wall-clock limit in seconds.
        max_memory_bytes: Optional resident/address-space limit in bytes.
    """
    if timeout is not None and timeout <= 0:
        raise ValueError("timeout must be greater than zero")
    if max_memory_bytes is not None and max_memory_bytes <= 0:
        raise ValueError("max_memory_bytes must be greater than zero")

    call_inputs = tuple(inputs)
    if timeout is None and max_memory_bytes is None:
        start = perf_counter()
        try:
            result = function(*call_inputs)
        except BaseException as exc:  # noqa: BLE001 - failures are part of the result
            return _failed_result(
                runtime=perf_counter() - start,
                expected=expected,
                error=f"{type(exc).__name__}: {exc}",
            )
        return RunResult(
            runtime=perf_counter() - start,
            result=result,
            expected=expected,
            result_is_expected=_matches_expected(result, expected),
            timed_out=False,
            memory_exceeded=False,
            error=None,
        )

    context = (
        mp.get_context("fork")
        if "fork" in mp.get_all_start_methods()
        else mp.get_context()
    )
    queue: mp.Queue = context.Queue()
    process = context.Process(
        target=_worker,
        args=(function, call_inputs, queue, max_memory_bytes),
    )

    start = perf_counter()
    deadline = None if timeout is None else start + timeout
    process.start()
    while process.is_alive():
        now = perf_counter()
        if deadline is not None and now >= deadline:
            _terminate_process_group(process)
            queue.close()
            return _failed_result(
                runtime=now - start,
                expected=expected,
                timed_out=True,
            )

        rss_bytes = (
            _process_group_rss_bytes(process.pid)
            if process.pid is not None and max_memory_bytes is not None
            else None
        )
        if (
            rss_bytes is not None
            and max_memory_bytes is not None
            and rss_bytes >= max_memory_bytes
        ):
            runtime = perf_counter() - start
            _terminate_process_group(process)
            queue.close()
            return _failed_result(
                runtime=runtime,
                expected=expected,
                memory_exceeded=True,
            )

        wait = MEMORY_POLL_INTERVAL_SECONDS
        if deadline is not None:
            wait = min(wait, max(0.0, deadline - now))
        process.join(wait)

    runtime = perf_counter() - start
    try:
        kind, payload, worker_runtime = queue.get(timeout=0.2)
    except Empty:
        return _failed_result(
            runtime=runtime,
            expected=expected,
            error=f"worker exited without a result (exit code {process.exitcode})",
        )
    finally:
        queue.close()

    if kind == "memory":
        return _failed_result(
            runtime=worker_runtime,
            expected=expected,
            memory_exceeded=True,
        )
    if kind == "error":
        return _failed_result(
            runtime=worker_runtime,
            expected=expected,
            error=str(payload),
        )

    return RunResult(
        runtime=worker_runtime,
        result=payload,
        expected=expected,
        result_is_expected=_matches_expected(payload, expected),
        timed_out=False,
        memory_exceeded=False,
        error=None,
    )


__all__ = ["RunResult", "run"]
