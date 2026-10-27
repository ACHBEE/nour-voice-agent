"""asyncio entry point that works on Windows.

psycopg's async mode can't run on Windows' default ProactorEventLoop (it raises InterfaceError,
and a connection pool silently retries forever). The selector loop works for psycopg and grpc.aio.
On Linux (Docker, CI) this is plain asyncio.run.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Coroutine
from typing import Any, TypeVar

T = TypeVar("T")


def run(coro: Coroutine[Any, Any, T]) -> T:
    loop_factory = asyncio.SelectorEventLoop if sys.platform == "win32" else None
    with asyncio.Runner(loop_factory=loop_factory) as runner:
        return runner.run(coro)
