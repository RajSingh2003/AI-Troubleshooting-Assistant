"""Hybrid retriever: Gemini embeddings (semantic) + keyword matching (lexical).

- Semantic: cosine similarity between the query embedding and each entry's
  embedding (gemini-embedding-001 by default). Entry embeddings are cached
  to data/embeddings.json and only recomputed when an entry's text changes.
- Lexical: overlap of query words with the entry's keywords/symptoms, plus
  exact phrase hits (e.g. "500", "upi pending"). Catches error codes that
  embeddings can blur.
- Offline mode (no API key): lexical only, so the demo still works.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

from .kb import ROOT, entry_search_text

CACHE_PATH = ROOT / "data" / "embeddings.json"
EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")

STOPWORDS = set("""
a an the is are was were be been being am i me my we our you your it its this that these those
to of in on at for from by with and or but not no so if then than too very can could would should
will just do does did doing have has had having what which who whom when where why how all any
some such only own same also again there here out up down over under into about after before
getting get got keeps keep showing shows show issue issues problem problems error errors
please help working work not unable cannot cant dont
""".split())


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


@dataclass
class Hit:
    entry: dict
    score: float          # combined score used for ranking / threshold
    semantic: float | None
    lexical: float


class Retriever:
    def __init__(self, entries: list[dict], client=None,
                 semantic_weight: float = 0.7, cache_path: Path = CACHE_PATH):
        self.entries = entries
        self.client = client  # OpenAI-compatible client (Gemini endpoint) or None (offline)
        self.semantic_weight = semantic_weight
        self.cache_path = Path(cache_path)
        self._lex_index = [self._lexical_fields(e) for e in entries]
        self.embeddings: dict[str, list[float]] = {}
        if self.client is not None:
            self._load_or_build_embeddings()

    @property
    def mode(self) -> str:
        return "hybrid (embeddings + keywords)" if self.client else "keyword only (offline)"

    # ---------- lexical ----------
    @staticmethod
    def _lexical_fields(e: dict) -> dict:
        strong = " ".join([e["title"], " ".join(e["keywords"])])
        weak = " ".join(e["symptoms"] + e.get("possible_causes", []) + [e["category"]])
        return {
            "strong": set(tokenize(strong)),
            "weak": set(tokenize(weak)),
            "phrases": [k.lower() for k in e["keywords"]],
        }

    def lexical_score(self, query: str, idx: int) -> float:
        q_tokens = set(tokenize(query))
        if not q_tokens:
            return 0.0
        f = self._lex_index[idx]
        strong = len(q_tokens & f["strong"])
        weak = len(q_tokens & (f["weak"] - f["strong"]))
        overlap = (strong + 0.5 * weak) / len(q_tokens)
        q_low = " " + re.sub(r"[^a-z0-9]+", " ", query.lower()) + " "
        phrase_hits = sum(1 for p in f["phrases"]
                          if " " + re.sub(r"[^a-z0-9]+", " ", p).strip() + " " in q_low)
        return min(1.0, 0.6 * overlap + 0.2 * phrase_hits)

    # ---------- semantic ----------
    def _embed(self, texts: list[str]) -> list[list[float]]:
        resp = self.client.embeddings.create(model=EMBED_MODEL, input=texts)
        return [d.embedding for d in resp.data]

    def _load_or_build_embeddings(self) -> None:
        cache = {}
        if self.cache_path.exists():
            try:
                cache = json.loads(self.cache_path.read_text())
            except json.JSONDecodeError:
                cache = {}
        if cache.get("model") != EMBED_MODEL:
            cache = {"model": EMBED_MODEL, "items": {}}
        items = cache.setdefault("items", {})

        todo = []
        for e in self.entries:
            text = entry_search_text(e)
            h = _hash(text)
            if items.get(e["id"], {}).get("hash") == h:
                self.embeddings[e["id"]] = items[e["id"]]["vector"]
            else:
                todo.append((e["id"], text, h))

        if todo:
            vectors = self._embed([t for _, t, _ in todo])
            for (eid, _, h), vec in zip(todo, vectors):
                items[eid] = {"hash": h, "vector": vec}
                self.embeddings[eid] = vec
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(cache))

    # ---------- search ----------
    def search(self, query: str, top_k: int = 3) -> list[Hit]:
        q_vec = self._embed([query])[0] if self.client else None
        hits = []
        for i, e in enumerate(self.entries):
            lex = self.lexical_score(query, i)
            if q_vec is not None:
                sem = _cosine(q_vec, self.embeddings[e["id"]])
                score = self.semantic_weight * sem + (1 - self.semantic_weight) * lex
            else:
                sem = None
                score = lex
            hits.append(Hit(entry=e, score=score, semantic=sem, lexical=lex))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]
