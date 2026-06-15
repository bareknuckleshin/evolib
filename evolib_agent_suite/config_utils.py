from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional


def normalize_library_config(library_config: Optional[Dict[str, Any]], *, default_path: str) -> Dict[str, Any]:
    """Return the effective nested EvoLib library policy config.

    The project originally accepted a flat ``library`` object.  New configs group
    policy knobs by EvoLib subsystem, but this helper keeps old config files
    working by mapping flat keys into their nested equivalents.
    """

    raw: Dict[str, Any] = deepcopy(library_config or {})
    retrieval = dict(raw.get("retrieval") or {})
    composition = dict(raw.get("composition") or {})
    ig = dict(raw.get("ig") or {})
    fig = dict(raw.get("fig") or {})
    consolidation = dict(raw.get("consolidation") or {})
    sampling = dict(raw.get("sampling") or {})
    storage = dict(raw.get("storage") or {})

    storage.setdefault("path", default_path)

    # Backward-compatible flat keys are treated as explicit overrides. This
    # keeps older test fixtures and programmatic overrides working even when
    # the base YAML has already migrated to nested sections.
    if "path" in raw:
        storage["path"] = raw["path"]
    retrieval["k_skills"] = raw.get("k_skills", retrieval.get("k_skills", 4))
    retrieval["k_insights"] = raw.get("k_insights", retrieval.get("k_insights", 4))
    retrieval["similarity_threshold"] = raw.get(
        "retrieval_similarity_threshold", retrieval.get("similarity_threshold", 0.05)
    )

    if "sample" in raw:
        retrieval["sampling_strategy"] = "weighted" if bool(raw["sample"]) else "topk"
    elif "sampling_strategy" not in retrieval:
        if "sample" in retrieval:
            retrieval["sampling_strategy"] = "weighted" if bool(retrieval["sample"]) else "topk"
        else:
            retrieval["sampling_strategy"] = "weighted"

    # Keep a boolean form for the current EvolvingLibrary.retrieve API.
    retrieval["sample"] = str(retrieval.get("sampling_strategy", "weighted")).lower() not in {
        "false",
        "none",
        "topk",
        "top_k",
        "deterministic",
    }

    consolidation["similarity_merge_threshold"] = raw.get(
        "similarity_merge_threshold", consolidation.get("similarity_merge_threshold", 0.88)
    )
    sampling.setdefault("strategy", retrieval.get("sampling_strategy", "weighted"))

    return {
        "retrieval": retrieval,
        "composition": composition,
        "ig": ig,
        "fig": fig,
        "consolidation": consolidation,
        "sampling": sampling,
        "storage": storage,
    }
