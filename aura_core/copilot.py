"""Aura Copilot — an on-device agentic RAG assistant.

The copilot answers driver questions ("why did you take over?", "what does Level 3 mean?",
"what should I do now?") grounded in a small local knowledge base (the owner's manual, the
AutoCare escalation ladder, Aura's safety policy, and DMS/ADAS regulation notes).

It is *agentic* in that it decides what context to bring: it always retrieves the most
relevant manual/policy chunks (RAG), and it can be handed the live driver-state + telemetry so
the answer is grounded in what is happening *right now*, not just static docs. Retrieval uses
nomic-embed-text; generation uses qwen2.5:7b — both local via [llm.py]. Answers cite sources.

Everything is best-effort and offline: if Ollama is unavailable it degrades to a keyword
search over the same chunks so the copilot still returns something useful.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from . import llm

log = logging.getLogger("aura-core")

_KB_DIR = os.path.join(os.path.dirname(__file__), "kb")
# ~120-word chunks keep each retrieved snippet focused enough to cite precisely.
_CHUNK_WORDS = 120
_TOP_K = 4


@dataclass
class Chunk:
    source: str          # human-readable doc title, e.g. "Owner's Manual"
    text: str
    embedding: Optional[List[float]] = None


def _title_from_filename(fname: str) -> str:
    stem = os.path.splitext(fname)[0]
    return stem.replace("_", " ").title()


def _chunk_markdown(text: str) -> List[str]:
    """Split a doc into readable chunks on blank lines, packing to ~_CHUNK_WORDS words.
    Markdown headings are kept with the paragraph they introduce so a citation reads well."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[str] = []
    buf: List[str] = []
    count = 0
    for p in paras:
        words = len(p.split())
        if count + words > _CHUNK_WORDS and buf:
            chunks.append("\n".join(buf))
            buf, count = [], 0
        buf.append(p)
        count += words
    if buf:
        chunks.append("\n".join(buf))
    return chunks


class Copilot:
    """Loads + embeds the KB once, then answers grounded questions."""

    def __init__(self) -> None:
        self.chunks: List[Chunk] = []
        self.embedded = False
        self._load()

    # ── indexing ─────────────────────────────────────────────────────
    def _load(self) -> None:
        if not os.path.isdir(_KB_DIR):
            log.warning("copilot: kb dir missing (%s)", _KB_DIR)
            return
        for fname in sorted(os.listdir(_KB_DIR)):
            if not fname.endswith(".md"):
                continue
            title = _title_from_filename(fname)
            try:
                with open(os.path.join(_KB_DIR, fname), "r", encoding="utf-8") as f:
                    body = f.read()
            except Exception as e:
                log.warning("copilot: could not read %s (%s)", fname, e)
                continue
            for c in _chunk_markdown(body):
                self.chunks.append(Chunk(source=title, text=c))
        log.info("copilot: loaded %d chunks from %d docs", len(self.chunks), len(set(c.source for c in self.chunks)))

    def ensure_embedded(self) -> None:
        """Embed all chunks on first use (lazy, so server startup stays instant)."""
        if self.embedded or not self.chunks:
            return
        ok = 0
        for c in self.chunks:
            if c.embedding is None:
                c.embedding = llm.embed(c.text)
                ok += 1 if c.embedding else 0
        self.embedded = any(c.embedding for c in self.chunks)
        log.info("copilot: embedded %d/%d chunks (Ollama %s)", ok, len(self.chunks),
                 "up" if self.embedded else "down — keyword fallback")

    # ── retrieval ────────────────────────────────────────────────────
    def retrieve(self, query: str, k: int = _TOP_K) -> List[Chunk]:
        self.ensure_embedded()
        qvec = llm.embed(query) if self.embedded else None
        if qvec:
            scored = [
                (llm.cosine(qvec, c.embedding), c) for c in self.chunks if c.embedding
            ]
            scored.sort(key=lambda t: t[0], reverse=True)
            return [c for _, c in scored[:k]]
        # Keyword fallback — still grounded, just cruder ranking.
        terms = [t for t in re.findall(r"\w+", query.lower()) if len(t) > 2]
        scored = [
            (sum(c.text.lower().count(t) for t in terms), c) for c in self.chunks
        ]
        scored.sort(key=lambda t: t[0], reverse=True)
        return [c for s, c in scored[:k] if s > 0] or self.chunks[:k]

    # ── answering ────────────────────────────────────────────────────
    def answer(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Answer a driver question, grounded in the KB (and optional live context).

        Returns {answer, sources[], grounded}. `context` may carry the current driver,
        drowsiness score, autocare level, and telemetry so 'what's happening now?' works.
        """
        hits = self.retrieve(query)
        sources = []
        seen = set()
        for c in hits:
            if c.source not in seen:
                sources.append(c.source)
                seen.add(c.source)

        kb_block = "\n\n".join(f"[{c.source}]\n{c.text}" for c in hits)
        live_block = _format_context(context) if context else ""

        system = (
            "You are Aura, an in-car driver co-pilot. Answer the driver's question briefly "
            "(2-4 sentences), in a calm, reassuring tone. Ground every claim ONLY in the "
            "provided KNOWLEDGE and LIVE STATE. If the answer isn't there, say what you do "
            "know and suggest the safe action. Never invent numbers. Cite the source doc "
            "names you used in square brackets at the end."
        )
        user = (
            (f"LIVE STATE:\n{live_block}\n\n" if live_block else "")
            + f"KNOWLEDGE:\n{kb_block}\n\n"
            + f"DRIVER QUESTION: {query}"
        )
        text = llm.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.3,
            num_predict=200,
        )
        if not text:
            # LLM down: return the most relevant snippet directly so we still help.
            text = (
                "Aura's language model is offline, but here's the most relevant guidance: "
                + (hits[0].text if hits else "No matching guidance found.")
            )
        return {"answer": text, "sources": sources, "grounded": bool(hits)}


def _format_context(ctx: Dict[str, Any]) -> str:
    """Render live driver/vehicle state as a compact block for grounding."""
    parts = []
    if ctx.get("driver"):
        parts.append(f"Driver: {ctx['driver']}")
    if ctx.get("score") is not None:
        parts.append(f"Drowsiness score: {ctx['score']}")
    if ctx.get("threshold") is not None:
        parts.append(f"This driver's personal threshold: {ctx['threshold']}")
    if ctx.get("level") is not None:
        parts.append(f"Current AutoCare level: {ctx['level']}")
    if ctx.get("speedKmh") is not None:
        parts.append(f"Vehicle speed: {ctx['speedKmh']} km/h")
    if ctx.get("scenario"):
        parts.append(f"Scenario: {ctx['scenario']}")
    return "\n".join(parts)
