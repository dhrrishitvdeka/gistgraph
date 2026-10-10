"""Training loop for compressors: distillation from the full-context teacher plus auxiliaries."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from gistgraph.config import Config
from gistgraph.data.schema import Example
from gistgraph.data.teacher_cache import TrainItem, to_tensor
from gistgraph.eval.run_learned import amp_dtype, dev_f1
from gistgraph.llm.frozen import FrozenLM
from gistgraph.train.losses import ce_loss, kd_loss, lr_at, recon_weight
from gistgraph.train.rate import RateController
from gistgraph.utils.fingerprint import atomic_torch_save, train_hash

RECON_QUESTION = "Repeat the context."
MAX_SKIPPED_STEPS = 20  # consecutive non-finite steps before giving up


def pad_ids(seqs: list[np.ndarray], pad_id: int, device) -> tuple[Tensor, Tensor]:
    """Right-pad token id arrays into ``[B, N]`` ids and a bool mask."""
    n = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), n), pad_id, dtype=torch.long)
    mask = torch.zeros(len(seqs), n, dtype=torch.bool)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.from_numpy(s)
        mask[i, : len(s)] = True
    return ids.to(device), mask.to(device)


def pad_targets(seqs: list[np.ndarray], device) -> tuple[Tensor, Tensor]:
    t = max(len(s) for s in seqs)
    out = torch.zeros(len(seqs), t, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = torch.from_numpy(s)
    return out.to(device), torch.tensor([len(s) for s in seqs], device=device)


def param_groups(module: torch.nn.Module, weight_decay: float) -> list[dict]:
    """AdamW groups: no weight decay on biases, norm weights or any other 1-D parameter."""
    decay, no_decay = [], []
    for name, p in module.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or name.endswith(".bias") else decay).append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def rng_state() -> dict:
    """Torch, CUDA, NumPy and Python RNG states, ``weights_only`` safe."""
    name, keys, pos, has_gauss, gauss = np.random.get_state()
    version, internal, gauss_next = random.getstate()
    state = {
        "torch": torch.get_rng_state(),
        "numpy": [name, torch.from_numpy(keys.astype(np.int64)), pos, has_gauss, gauss],
        "python": [version, list(internal), gauss_next],
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict) -> None:
    torch.set_rng_state(state["torch"].cpu())
    name, keys, pos, has_gauss, gauss = state["numpy"]
    np.random.set_state((name, keys.cpu().numpy().astype(np.uint32), pos, has_gauss, gauss))
    version, internal, gauss_next = state["python"]
    random.setstate((version, tuple(internal), gauss_next))
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


class Trainer:
    def __init__(
        self,
        cfg: Config,
        lm: FrozenLM,
        compressor,
        items: list[TrainItem],
        dev_examples: list[Example] | None = None,
        force: bool = False,
    ):
        if not items:
            raise ValueError("no training items: the teacher cache and examples share no ids")
        self.cfg, self.lm, self.compressor = cfg, lm, compressor
        self.force, self.config_hash = force, train_hash(cfg)
        self.items, self.dev = items, dev_examples or []
        self.device = lm.device
        self.out = Path(cfg.out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        t = cfg.train
        self.opt = torch.optim.AdamW(param_groups(compressor, 0.01), lr=t.lr, betas=(0.9, 0.95))
        self.use_amp = bool(t.amp and self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp and amp_dtype() == torch.float16)
        self.step = 0
        self.rate = RateController(cfg.loss.rate_dual_lr, cfg.loss.rate_quad)
        self._rate_updates: list[tuple[float, float]] = []  # applied only after a finite step
        self.skipped = 0

    # ---- data ----
    def _batch(self, micro_idx: int) -> list[TrainItem]:
        b = self.cfg.train.batch
        n = len(self.items)
        epoch, pos = divmod(micro_idx * b, n)
        perm = np.random.default_rng([self.cfg.seed, epoch]).permutation(n)
        idx = [perm[(pos + j) % n] for j in range(b)]
        return [self.items[i] for i in idx]

    def _ratio(self, micro_idx: int) -> float:
        rng = np.random.default_rng([self.cfg.seed, 1, micro_idx])
        return float(rng.choice(self.cfg.train.ratios))

    def _chunks(self) -> int:
        s = self.cfg.compressor.streaming
        return s.chunks if s.enabled else 1

    # ---- losses ----
    def _losses(self, batch: list[TrainItem], ratio: float, recon_w: float) -> dict[str, Tensor]:
        cfg, lm = self.cfg, self.lm
        pad_id = lm.tokenizer.pad_token_id
        ids, mask = pad_ids([it.ctx_ids for it in batch], pad_id, self.device)
        h = lm.token_features(ids, mask, cfg.llm.feature_layer)
        with torch.autocast(self.device.type, dtype=amp_dtype(), enabled=self.use_amp):
            mem = self.compressor(h, mask, ratio, chunks=self._chunks())
        prefixes = [mem.prefix(i) for i in range(len(batch))]
        questions = [it.ex.question for it in batch]
        out: dict[str, Tensor] = {}

        def answer_pass(qs, targets_np):
            targets, lens = pad_targets(targets_np, self.device)
            tgt_embeds = [to_tensor(t, self.device) for t in targets_np]
            embeds, amask, starts = lm.build_answer_inputs(qs, prefixes, tgt_embeds)
            logits, valid = lm.answer_logits(embeds, amask, starts, lens)
            return logits, valid, targets

        # distillation towards the teacher's full-context answer distribution
        logits, valid, _ = answer_pass(questions, [it.ans_ids for it in batch])
        t = logits.shape[1]
        idx = torch.zeros(len(batch), t, batch[0].topk_idx.shape[1], dtype=torch.long)
        val = torch.zeros(len(batch), t, batch[0].topk_idx.shape[1])
        for i, it in enumerate(batch):
            n = len(it.ans_ids)
            idx[i, :n] = torch.from_numpy(it.topk_idx)
            val[i, :n] = torch.from_numpy(it.topk_logits)
        out["kd"] = kd_loss(
            logits, idx.to(self.device), val.to(self.device), valid, cfg.loss.kd_temp
        )

        if recon_w > 0:  # auxiliary: rebuild the start of the context from the memory
            r = cfg.train.recon_tokens
            recon_targets = [it.ctx_ids[:r] for it in batch]
            logits, valid, targets = answer_pass([RECON_QUESTION] * len(batch), recon_targets)
            out["recon"] = ce_loss(logits, targets, valid)

        if cfg.loss.qa_ce_weight > 0:  # auxiliary: gold answers (QA probe)
            logits, valid, targets = answer_pass(questions, [it.gold_ids for it in batch])
            out["qa"] = ce_loss(logits, targets, valid)

        out["total"] = out["kd"]
        if "recon" in out:
            out["total"] = out["total"] + recon_w * out["recon"]
        if "qa" in out:
            out["total"] = out["total"] + cfg.loss.qa_ce_weight * out["qa"]
        weights = {"lb": cfg.loss.lb_weight, "rate": 1.0, "edge_l0": cfg.loss.edge_l0}
        for name, term in self._regularisers(mem, ratio).items():
            out[name] = term
            if name in weights:
                out["total"] = out["total"] + weights[name] * term
        return out

    def _regularisers(self, mem, ratio: float) -> dict[str, Tensor]:
        """Variant-specific terms read from ``mem.aux``.

        ``lb`` is the router load-balance loss, ``rate`` the Lagrangian rate penalty on the open
        fraction of nodes (active only after ``rate_start_frac`` of training), and ``frac`` /
        ``lambda`` are logged diagnostics.
        """
        aux, out = mem.aux, {}
        if "lb" in aux:
            out["lb"] = aux["lb"]
        if "exp_active" in aux:
            frac = (aux["exp_active"] / aux["n_tokens"]).mean()
            out["frac"] = frac.detach()
            out["lambda"] = torch.tensor(self.rate.multiplier(ratio))
            if self.step >= self.cfg.loss.rate_start_frac * self.cfg.train.steps:
                out["rate"] = self.rate.penalty(ratio, frac)
                self._rate_updates.append((ratio, float(frac.detach())))
        if "exp_edges" in aux:
            out["edge_l0"] = (aux["exp_edges"] / aux["n_tokens"]).mean()
        for key in ("active", "mean_degree", "edge_entropy", "edge_mass"):  # diagnostics only
            if key in aux:
                out[key] = aux[key].float().mean().detach()
        return out

    # ---- checkpoints ----
    def _check_finite(self) -> None:
        bad = [n for n, p in self.compressor.named_parameters() if not torch.isfinite(p).all()]
        if bad:
            raise RuntimeError(f"refusing to save: non-finite parameters {bad[:5]}")

    def _save(self, name: str = "ckpt.pt") -> None:
        self._check_finite()
        state = {
            "compressor": self.compressor.state_dict(),
            "opt": self.opt.state_dict(),
            "scaler": self.scaler.state_dict(),
            "rate": self.rate.state_dict(),
            "step": self.step,
            "config_hash": self.config_hash,
            "rng": rng_state(),
        }
        atomic_torch_save(state, self.out / name)

    def _resume(self) -> None:
        path = self.out / "ckpt.pt"
        if not path.exists():
            return
        state = torch.load(path, map_location=self.device, weights_only=True)
        stored = state.get("config_hash")
        if stored != self.config_hash and not self.force:
            raise RuntimeError(
                f"{path} was written with config hash {stored}, the current config hashes to "
                f"{self.config_hash}; use a fresh out_dir or pass force=True to resume anyway"
            )
        self.compressor.load_state_dict(state["compressor"])
        self.opt.load_state_dict(state["opt"])
        self.scaler.load_state_dict(state["scaler"])
        self.rate.load_state_dict(state["rate"])
        self.step = state["step"]
        if "rng" in state:
            set_rng_state(state["rng"])
        print(f"resumed from step {self.step}", flush=True)

    def _log(self, record: dict) -> None:
        with (self.out / "train_log.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    # ---- loop ----
    def _skip_step(self) -> None:
        """Drop a step whose loss or gradient norm is non-finite; abort if it keeps happening."""
        self.opt.zero_grad(set_to_none=True)
        if self.scaler.is_enabled():  # fp16: let the scaler back off its scale
            self.scaler.update(self.scaler.get_scale() / 2)
        self.skipped += 1
        self._log({"step": self.step, "skipped": True})
        print(f"step {self.step}: non-finite loss or grad norm, update skipped", flush=True)
        if self.skipped >= MAX_SKIPPED_STEPS:
            raise RuntimeError(f"{self.skipped} consecutive non-finite steps; aborting")

    def train(self) -> Path:
        cfg, t = self.cfg, self.cfg.train
        self._resume()
        self.compressor.train()
        start = time.time()
        while self.step < t.steps:
            lr = lr_at(self.step, t.steps, t.lr, t.warmup)
            for g in self.opt.param_groups:
                g["lr"] = lr
            w = recon_weight(self.step, t.steps, cfg.loss.recon_weight, cfg.loss.recon_anneal_frac)
            acc: dict[str, float] = {}
            finite = True
            self._rate_updates.clear()
            for a in range(t.grad_accum):
                micro = self.step * t.grad_accum + a
                ratio = self._ratio(micro)
                losses = self._losses(self._batch(micro), ratio, w)
                if not torch.isfinite(losses["total"]).all():
                    finite = False
                    break
                self.scaler.scale(losses["total"] / t.grad_accum).backward()
                for k, v in losses.items():
                    acc[k] = acc.get(k, 0.0) + float(v.detach()) / t.grad_accum
            if finite:
                self.scaler.unscale_(self.opt)
                gnorm = torch.nn.utils.clip_grad_norm_(self.compressor.parameters(), 1.0)
                finite = bool(torch.isfinite(gnorm))
            if not finite:
                self._skip_step()
                self.step += 1  # move past the bad batch rather than retrying it
                continue
            self.skipped = 0
            self.scaler.step(self.opt)
            self.scaler.update()
            self.opt.zero_grad(set_to_none=True)
            for r, f in self._rate_updates:
                self.rate.update(r, f)
            self.step += 1
            if self.step % 10 == 0 or self.step == t.steps:
                rec = {"step": self.step, "lr": lr, "grad_norm": float(gnorm), **acc}
                rec["elapsed_s"] = round(time.time() - start, 1)
                self._log(rec)
                print(" ".join(f"{k}={v:.4g}" for k, v in rec.items()), flush=True)
            if self.step % t.ckpt_every == 0:
                self._save()
            if self.dev and self.step % t.eval_every == 0:
                f1 = dev_f1(self.lm, self.compressor, cfg, self.dev, 4.0)
                self._log({"step": self.step, "dev_f1_at_4x": f1})
                print(f"dev F1 @4x: {f1:.4f}", flush=True)
        self._save()
        final = self.out / "compressor.pt"
        payload = {
            "state_dict": self.compressor.state_dict(),
            "config_hash": self.config_hash,
            "step": self.step,
        }
        atomic_torch_save(payload, final)
        return final
