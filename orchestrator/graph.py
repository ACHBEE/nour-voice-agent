"""The LangGraph state machine and a small facade the voice agent calls once per caller turn.

    START -> route -+-> retrieve -> answer -> END      clinic_info
                    +-> answer -> END                  smalltalk
                    +-> booking -> END                 book_appointment (multi-turn form)
                    +-> callback -> END                human_callback (multi-turn form)
                    +-> safety -> END                  emergency / medical_advice (fixed phrases)

State persists per call in the checkpointer (thread_id = call id), which is what lets a booking
be collected over several turns and survive a clinic question asked in the middle of it.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from datetime import date

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from orchestrator import nodes
from orchestrator.deps import Deps, RagSearch, Store
from orchestrator.llm import LLMClient
from orchestrator.schedule import clinic_today
from orchestrator.state import CallState


def build_graph(checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph:
    g = StateGraph(CallState, context_schema=Deps)
    g.add_node("route", nodes.route)
    g.add_node("retrieve", nodes.retrieve)
    g.add_node("answer", nodes.answer)
    g.add_node("booking", nodes.booking)
    g.add_node("callback", nodes.callback)
    g.add_node("safety", nodes.safety)

    g.add_edge(START, "route")
    g.add_conditional_edges(
        "route", nodes.next_step, ["retrieve", "answer", "booking", "callback", "safety"]
    )
    g.add_edge("retrieve", "answer")
    for leaf in ("answer", "booking", "callback", "safety"):
        g.add_edge(leaf, END)
    return g.compile(checkpointer=checkpointer or InMemorySaver())


class Orchestrator:
    """One instance per process; each call is a separate thread in the checkpointer."""

    def __init__(
        self,
        llm: LLMClient,
        rag: RagSearch,
        store: Store,
        *,
        today: Callable[[], date] = clinic_today,
        router_llm: LLMClient | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
    ):
        self.llm, self.rag, self.store, self.today = llm, rag, store, today
        self.router_llm = router_llm
        self.checkpointer = checkpointer or InMemorySaver()
        self.graph = build_graph(self.checkpointer)

    @staticmethod
    def _config(call_id: str) -> dict:
        return {"configurable": {"thread_id": call_id}}

    async def turn(self, call_id: str, text: str) -> AsyncIterator[str]:
        """Run one caller turn; yields Nour's reply as text chunks as soon as they exist."""
        deps = Deps(
            self.llm,
            self.rag,
            self.store,
            call_id=call_id,
            today=self.today,
            router_llm=self.router_llm,
        )
        turn_input: CallState = {
            "user_text": text,
            "messages": [{"role": "user", "content": text}],
        }
        async for chunk in self.graph.astream(
            turn_input, self._config(call_id), context=deps, stream_mode="custom"
        ):
            yield chunk["text"]

    async def reply(self, call_id: str, text: str) -> str:
        return "".join([c async for c in self.turn(call_id, text)])

    async def state(self, call_id: str) -> CallState:
        """State after the last turn: intent, timings, chunks, open forms..."""
        return (await self.graph.aget_state(self._config(call_id))).values

    async def end_call(self, call_id: str) -> None:
        await self.checkpointer.adelete_thread(call_id)
