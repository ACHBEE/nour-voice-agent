"""Load the clinic knowledge base (markdown files with a small front-matter header)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "clinic_docs"

Lang = Literal["ar", "fr", "en"]


class ClinicDoc(BaseModel):
    doc_id: str = Field(..., pattern=r"^[a-z0-9_]+$")
    title: str = Field(..., min_length=3)
    lang: Lang
    category: str
    dialect: str | None = None
    body: str = Field(..., min_length=50)
    source_path: str

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.body.encode("utf-8")).hexdigest()


def parse_doc(path: Path) -> ClinicDoc:
    """Parse `---`-delimited `key: value` front matter followed by a markdown body."""
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        raise ValueError(f"{path.name}: missing front matter")
    _, header, body = text.split("---", 2)
    meta: dict[str, str] = {}
    for line in header.strip().splitlines():
        key, sep, value = line.partition(":")
        if not sep:
            raise ValueError(f"{path.name}: bad front-matter line {line!r}")
        meta[key.strip()] = value.strip()
    return ClinicDoc(**meta, body=body.strip(), source_path=path.name)


def load_docs(docs_dir: Path = DOCS_DIR) -> list[ClinicDoc]:
    return [parse_doc(p) for p in sorted(docs_dir.glob("*.md"))]
