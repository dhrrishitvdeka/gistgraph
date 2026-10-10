"""Precomputed teacher outputs for distillation.

The teacher is the frozen LLM reading the *full* context. Running it once per training example and
storing its greedy answer plus the top-k logits at each answer step means training never has to
run the long full-context forward pass again, which is the main compute saving on small GPUs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from gistgraph.data.schema import Example


@dataclass
class TrainItem:
    """One distillation example, ready for batching."""

    ex: Example
    ctx_ids: np.ndarray  # context tokens, already cut to max_ctx_tokens
    ans_ids: np.ndarray  # teacher's greedy answer tokens (including the end token)
    topk_idx: np.ndarray  # [T, k] teacher top-k token ids per answer step
    topk_logits: np.ndarray  # [T, k] their logits
    gold_ids: np.ndarray  # first gold answer tokens plus the end token


def build_teacher_cache(
    lm,
    examples: list[Example],
    path: str | Path,
    topk: int = 64,
    max_new_tokens: int = 24,
    batch_size: int = 8,
) -> None:
    """Run the teacher on ``examples`` and save ids and top-k logits to ``path`` (an ``.npz``)."""
    order = sorted(range(len(examples)), key=lambda i: len(examples[i].context))
    keys, offsets, ids_flat, idx_flat, logit_flat = [], [0], [], [], []
    for s in range(0, len(order), batch_size):
        batch = [examples[i] for i in order[s : s + batch_size]]
        embeds, mask = lm.build_inputs(
            [e.question for e in batch], [None] * len(batch), contexts=[e.context for e in batch]
        )
        ids, lens, idx, logits = lm.generate_scored(embeds, mask, max_new_tokens, topk)
        for j, e in enumerate(batch):
            n = int(lens[j])
            keys.append(e.id)
            ids_flat.append(ids[j, :n].cpu().numpy().astype(np.int32))
            idx_flat.append(idx[j, :n].cpu().numpy().astype(np.int32))
            logit_flat.append(logits[j, :n].cpu().numpy().astype(np.float16))
            offsets.append(offsets[-1] + n)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")  # np.savez appends .npz to other names
    np.savez(
        tmp,
        keys=np.array(keys),
        offsets=np.array(offsets, dtype=np.int64),
        ids=np.concatenate(ids_flat),
        topk_idx=np.concatenate(idx_flat),
        topk_logits=np.concatenate(logit_flat),
    )
    os.replace(tmp, path)


def load_teacher_cache(path: str | Path) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return ``{example_id: (answer_ids, topk_idx, topk_logits)}``."""
    with np.load(path, allow_pickle=False) as z:  # each z[...] access re-reads the archive
        keys, off = z["keys"], z["offsets"]
        ids, idx, logits = z["ids"], z["topk_idx"], z["topk_logits"]
    return {
        str(k): (ids[off[i] : off[i + 1]], idx[off[i] : off[i + 1]], logits[off[i] : off[i + 1]])
        for i, k in enumerate(keys)
    }


def make_train_items(
    tokenizer, examples: list[Example], teacher: dict, max_ctx_tokens: int, eos_id: int
) -> list[TrainItem]:
    """Join examples with their cached teacher outputs; tokenise contexts and gold answers."""
    items, dropped = [], 0
    for ex in examples:
        if ex.id not in teacher:
            dropped += 1
            continue
        ans_ids, idx, logits = teacher[ex.id]
        ctx = tokenizer.encode(ex.context, add_special_tokens=False)[:max_ctx_tokens]
        gold = tokenizer.encode(ex.answers[0], add_special_tokens=False) + [eos_id]
        items.append(
            TrainItem(
                ex,
                np.array(ctx, dtype=np.int64),
                ans_ids.astype(np.int64),
                idx.astype(np.int64),
                logits.astype(np.float32),
                np.array(gold, dtype=np.int64),
            )
        )
    if dropped:
        print(f"warning: {dropped} examples have no teacher output and were dropped", flush=True)
    return items


def to_tensor(arr: np.ndarray, device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(arr)).to(device)
