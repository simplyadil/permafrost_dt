"""Simple helpers for launching service components in separate processes."""

from __future__ import annotations

from multiprocessing import Process, get_context
from multiprocessing.queues import Queue
from typing import Any, Callable, Dict, Optional


def start_as_daemon(
    component_starter_function: Callable[..., None],
    *,
    process_name: Optional[str] = None,
    kwargs: Optional[Dict[str, Any]] = None,
) -> Process:
    """Start ``component_starter_function`` inside a daemon process.

    The function receives an ``ok_queue`` keyword argument to report when it has
    finished initialisation.  The queue should receive a truthy value to signal
    success; otherwise the launcher will block.
    """

    if kwargs is None:
        kwargs = {}

    # Use the 'fork' start method so nested/local callables (closures) do not
    # need to be pickled. On some Python builds the default context may be
    # 'forkserver' or 'spawn', which requires targets to be importable
    # top-level callables. For this launcher we run on Linux where 'fork' is
    # available and acceptable.
    ctx = get_context("fork")

    ok_queue: Queue[Any] = Queue(ctx=ctx)
    kwargs.setdefault("ok_queue", ok_queue)

    resolved_name = process_name or component_starter_function.__name__
    process = ctx.Process(target=component_starter_function, kwargs=kwargs, name=resolved_name, daemon=True)
    process.start()

    status = ok_queue.get()
    print(f"{resolved_name}... {status}")

    return process


__all__ = ["start_as_daemon"]
