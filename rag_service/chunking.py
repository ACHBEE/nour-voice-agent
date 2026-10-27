"""Section-based chunking for the clinic docs.

One chunk per markdown section, prefixed with "<doc title> › <section heading>" so every chunk
carries its own context (a bare "2000 DZD." is useless without "Tarifs › Consultations").
Sections longer than MAX_CHARS are split on paragraph boundaries.
"""

from __future__ import annotations

import re

from pydantic import BaseModel

from rag_service.docs import ClinicDoc

MAX_CHARS = 900  # ~150-250 tokens: one answerable fact cluster, small enough for a voice reply
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


class DocChunk(BaseModel):
    chunk_index: int
    section: str
    content: str  # contextual header + section text: this is what gets embedded and indexed
    lang: str
    token_count: int


def _sections(body: str, default_heading: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, list[str]]] = [(default_heading, [])]
    for line in body.splitlines():
        m = _HEADING.match(line.strip())
        if m:
            sections.append((m.group(2).strip(), []))
        else:
            sections[-1][1].append(line)
    out = []
    for heading, lines in sections:
        text = "\n".join(lines).strip()
        if text:
            out.append((heading, text))
    return out


def _split_long(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    parts, cur = [], ""
    for para in re.split(r"\n\s*\n", text):
        if cur and len(cur) + len(para) + 2 > max_chars:
            parts.append(cur)
            cur = para
        else:
            cur = f"{cur}\n\n{para}" if cur else para
    if cur:
        parts.append(cur)
    return parts


def chunk_doc(doc: ClinicDoc, max_chars: int = MAX_CHARS) -> list[DocChunk]:
    chunks: list[DocChunk] = []
    for heading, text in _sections(doc.body, default_heading=doc.title):
        for piece in _split_long(text, max_chars):
            header = doc.title if heading == doc.title else f"{doc.title} › {heading}"
            content = f"{header}\n{piece}"
            chunks.append(
                DocChunk(
                    chunk_index=len(chunks),
                    section=heading,
                    content=content,
                    lang=doc.lang,
                    token_count=len(content.split()),
                )
            )
    return chunks
