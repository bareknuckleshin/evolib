from __future__ import annotations

import json
import math
import os
import random
import re
import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np


def load_config(path: str | os.PathLike[str]) -> Dict[str, Any]:
    path = str(path)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if path.endswith(".json"):
        return json.loads(text)
    try:
        import yaml

        return yaml.safe_load(text) or {}
    except Exception as exc:
        raise RuntimeError(
            f"Could not load {path}. Install PyYAML or use JSON config. Original error: {exc}"
        ) from exc


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def append_jsonl(path: str | os.PathLike[str], record: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: str | os.PathLike[str]) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_\-]+", (text or "").lower())


def stable_hash(value: str) -> int:
    return int(hashlib.md5(value.encode("utf-8")).hexdigest(), 16)


def hashed_embedding(text: str, dim: int = 384) -> List[float]:
    """Small dependency-free embedding for retrieval.

    It is intentionally simple. Swap this function with a sentence-transformer or
    API embedding client if you want stronger semantic retrieval.
    """
    vec = np.zeros(dim, dtype=np.float32)
    tokens = tokenize(text)
    if not tokens:
        return vec.tolist()
    for tok in tokens:
        h = stable_hash(tok)
        idx = h % dim
        sign = -1.0 if ((h >> 9) & 1) else 1.0
        vec[idx] += sign
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return vec.tolist()


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(va, vb) / denom)


def weighted_sample_without_replacement(
    items: Sequence[Any], weights: Sequence[float], k: int, rng: random.Random
) -> List[Any]:
    pool = list(items)
    ws = [max(float(w), 1e-8) for w in weights]
    out: List[Any] = []
    for _ in range(min(k, len(pool))):
        total = sum(ws)
        r = rng.random() * total
        upto = 0.0
        chosen = len(pool) - 1
        for i, w in enumerate(ws):
            upto += w
            if upto >= r:
                chosen = i
                break
        out.append(pool.pop(chosen))
        ws.pop(chosen)
    return out


def extract_json_block(text: str) -> Optional[Any]:
    """Parse the first JSON object/list from model text."""
    text = (text or "").strip()
    if not text:
        return None
    candidates = [text]
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        candidates.insert(0, fence.group(1).strip())
    bracket_positions = []
    for ch in ["{", "["]:
        idx = text.find(ch)
        if idx >= 0:
            bracket_positions.append(idx)
    if bracket_positions:
        candidates.append(text[min(bracket_positions) :])
    for c in candidates:
        try:
            return json.loads(c)
        except Exception:
            continue
    return None


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))
