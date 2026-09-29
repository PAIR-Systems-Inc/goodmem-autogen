"""SDK client ownership and cancellation-aware awaiting."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import TypeVar, cast

from autogen_core import CancellationToken
from goodmem import AsyncGoodmem
from typing_extensions import Self

from ._typing import AsyncGoodmemClient


T = TypeVar("T")


async def run_cancellable(
    awaitable: Awaitable[T],
    cancellation_token: CancellationToken | None = None,
) -> T:
    """Await ``awaitable``, honouring an AutoGen cancellation token.

    ``CancellationToken.link_future`` cancels the wrapped future when the
    token fires; this is the pattern AutoGen's own model clients use. Note
    the token exposes ``is_cancelled()`` and ``link_future()`` — there is no
    ``.cancelled`` attribute, despite what some ecosystem code assumes.
    """
    future = asyncio.ensure_future(awaitable)
    if cancellation_token is not None:
        if cancellation_token.is_cancelled():
            future.cancel()
            raise asyncio.CancelledError("operation cancelled before it was issued")
        cancellation_token.link_future(future)
    return await future


class GoodMemConnection:
    """Owns an ``AsyncGoodmem`` client, or borrows a caller-supplied one.

    An injected client keeps its own server, credentials and TLS settings;
    this class never closes it. Otherwise one client is created lazily and
    closed by :meth:`close`. There is no process-wide client cache.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        verify_ssl: bool | str = True,
        timeout: float = 60.0,
        client: AsyncGoodmem | None = None,
    ) -> None:
        self._base_url = base_url
        self._api_key = api_key
        self._verify_ssl = verify_ssl
        self._timeout = timeout
        self._injected = client
        self._owned: AsyncGoodmem | None = None

    @property
    def owns_client(self) -> bool:
        return self._injected is None

    def client(self) -> AsyncGoodmemClient:
        if self._injected is not None:
            return cast(AsyncGoodmemClient, self._injected)
        if self._owned is None:
            if not self._base_url or not self._api_key:
                raise ValueError(
                    "GoodMem base_url and api_key are required when no client is injected."
                )
            self._owned = AsyncGoodmem(
                base_url=self._base_url.rstrip("/"),
                api_key=self._api_key,
                verify=self._verify_ssl,
                timeout=self._timeout,
            )
        return cast(AsyncGoodmemClient, self._owned)

    async def close(self) -> None:
        if self._owned is not None:
            await self._owned.close()
            self._owned = None

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()
