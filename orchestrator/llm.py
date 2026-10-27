"""LLM access for the graph nodes, behind a two-method interface.

The default implementation calls LiveKit Inference (same gateway and keys as the voice agent),
so the orchestrator needs no extra provider key. Tests swap in a scripted fake.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Protocol, TypeVar

from livekit.agents import inference, llm
from pydantic import BaseModel

from orchestrator.config import LLM_MODEL
from orchestrator.state import Msg

T = TypeVar("T", bound=BaseModel)


class LLMClient(Protocol):
    async def structured(self, system: str, history: Sequence[Msg], schema: type[T]) -> T:
        """One completion constrained to `schema` (strict JSON schema), parsed and validated."""
        ...

    def stream(self, system: str, history: Sequence[Msg]) -> AsyncIterator[str]:
        """Free-text completion, streamed as text deltas."""
        ...


def _chat_ctx(system: str, history: Sequence[Msg]) -> llm.ChatContext:
    ctx = llm.ChatContext.empty()
    ctx.add_message(role="system", content=system)
    for m in history:
        ctx.add_message(role=m["role"], content=m["content"])
    return ctx


class LiveKitLLM:
    def __init__(self, model: str = LLM_MODEL):
        self.model = model
        self._llm = inference.LLM(model=model)

    async def structured(self, system: str, history: Sequence[Msg], schema: type[T]) -> T:
        resp = await self._llm.chat(
            chat_ctx=_chat_ctx(system, history),
            response_format=schema,
            extra_kwargs={"temperature": 0},
        ).collect()
        return schema.model_validate_json(resp.text)

    async def stream(self, system: str, history: Sequence[Msg]) -> AsyncIterator[str]:
        stream = self._llm.chat(chat_ctx=_chat_ctx(system, history))
        async for delta in stream.to_str_iterable():
            yield delta

    async def aclose(self) -> None:
        await self._llm.aclose()
