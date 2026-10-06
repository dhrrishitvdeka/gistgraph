"""Structure probes: do the induced edges mean anything?

For HotpotQA we know which two paragraphs hold the supporting facts. A node's *home paragraph* is
the paragraph that contributed most of the write weight it received. If edge induction learned
something real, edges should connect the homes of the two supporting paragraphs more often than a
content-blind graph would. The score is the observed fraction of edge weight between those two
paragraphs divided by the fraction expected if edges were placed uniformly at random between the
same active nodes (1.0 means no better than random).
"""

from __future__ import annotations

import numpy as np
import torch

from gistgraph.data.datasets import CONTEXT_SEP
from gistgraph.data.schema import Example
from gistgraph.eval.analysis import mean_ci


def paragraph_token_map(tokenizer, ex: Example) -> np.ndarray | None:
    """Paragraph index of every context token, or None when paragraphs are unavailable."""
    if len(ex.paragraphs) < 2 or CONTEXT_SEP.join(ex.paragraphs) != ex.context:
        return None
    starts = np.cumsum([0] + [len(p) + len(CONTEXT_SEP) for p in ex.paragraphs[:-1]])
    enc = tokenizer(ex.context, add_special_tokens=False, return_offsets_mapping=True)
    first_char = np.array([s for s, _ in enc["offset_mapping"]])
    return np.searchsorted(starts, first_char, side="right") - 1


def node_home_paragraph(write: np.ndarray, token_para: np.ndarray, n_para: int):
    """Home paragraph per node and whether it received any weight. ``write``: ``[M, K]``."""
    onehot = np.eye(n_para)[token_para[: write.shape[0]]]  # [M, P]
    mass = write.T @ onehot  # [K, P]
    return mass.argmax(1), mass.sum(1) > 1e-6


def edge_enrichment(
    adj: np.ndarray, home: np.ndarray, usable: np.ndarray, supporting: list[int]
) -> dict:
    """Bridge-edge enrichment for one document.

    ``adj``: ``[R, K, K]`` edge weights; ``home``: home paragraph per node; ``usable``: nodes that
    are active and received weight; ``supporting``: indices of the supporting paragraphs.
    """
    w = adj.sum(0) * np.outer(usable, usable)
    np.fill_diagonal(w, 0.0)
    sup = np.isin(home, supporting)
    bridge = np.outer(sup, sup) & (home[:, None] != home[None, :])
    same = home[:, None] == home[None, :]
    pair = np.outer(usable, usable) & ~np.eye(len(home), dtype=bool)
    total, n_pairs = w.sum(), pair.sum()
    if total <= 0 or n_pairs == 0:
        return {"enrichment": float("nan"), "within_enrichment": float("nan")}
    expected_bridge = (bridge & pair).sum() / n_pairs
    expected_same = (same & pair).sum() / n_pairs
    return {
        "bridge_fraction": float((w * bridge).sum() / total),
        "expected_bridge": float(expected_bridge),
        "enrichment": (
            float((w * bridge).sum() / total / expected_bridge)
            if expected_bridge > 0
            else float("nan")
        ),
        "within_enrichment": (
            float((w * same).sum() / total / expected_same) if expected_same > 0 else float("nan")
        ),
    }


def summarise(per_example: list[dict]) -> dict:
    """Mean enrichment with a bootstrap interval, over examples where it is defined."""
    out = {"n_examples": len(per_example)}
    for key in ("enrichment", "within_enrichment"):
        vals = [r[key] for r in per_example if not np.isnan(r.get(key, float("nan")))]
        mean, lo, hi = mean_ci(vals)
        out[key] = {"mean": mean, "lo": lo, "hi": hi, "n": len(vals)}
    vals = [r["enrichment"] for r in per_example if not np.isnan(r.get("enrichment", float("nan")))]
    out["fraction_above_random"] = float(np.mean([v > 1.0 for v in vals])) if vals else float("nan")
    return out


@torch.no_grad()
def probe_model(lm, compressor, cfg, examples: list[Example], ratio: float) -> dict:
    """Run the bridge-edge probe on HotpotQA-style examples with known supporting paragraphs."""
    compressor.eval()
    per_example = []
    for ex in examples:
        token_para = paragraph_token_map(lm.tokenizer, ex)
        if token_para is None or len(ex.supporting) != 2:
            continue
        ids = lm.encode_text(ex.context)[None]
        if ids.shape[1] > cfg.data.max_ctx_tokens:
            continue  # cut contexts would misalign paragraph bookkeeping
        mask = torch.ones_like(ids)
        h = lm.token_features(ids, mask, cfg.llm.feature_layer)
        mem = compressor(h, mask.bool(), ratio)
        aux = mem.aux
        if "adj" not in aux or "write" not in aux:
            raise ValueError("the probe needs a graph model with learned edges")
        write = aux["write"][0].float().cpu().numpy()
        home, has_mass = node_home_paragraph(write, token_para, len(ex.paragraphs))
        active = aux["gate"][0].cpu().numpy() > 0
        per_example.append(
            edge_enrichment(
                aux["adj"][0].float().cpu().numpy(), home, active & has_mass, ex.supporting
            )
        )
    return summarise(per_example)
