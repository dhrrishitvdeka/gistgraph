"""Training loop for compressors: distillation from the full-context teacher plus auxiliaries."""

from __future__ import annotations

import json
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

RECON_QUESTION = "Repeat the context."


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


class Trainer:
    def __init__(
        self,
        cfg: Config,
        lm: FrozenLM,
        compressor,
        items: list[TrainItem],
        dev_examples: list[Example] | None = None,
    ):
        self.cfg, self.lm, self.compressor = cfg, lm, compressor
        self.items, self.dev = items, dev_examples or []
        self.device = lm.device
        self.out = Path(cfg.out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        t = cfg.train
        self.opt = torch.optim.AdamW(
            compressor.parameters(), lr=t.lr, betas=(0.9, 0.95), weight_decay=0.01
        )
        self.use_amp = bool(t.amp and self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp and amp_dtype() == torch.float16)
        self.step = 0
        self.rate = RateController(cfg.loss.rate_dual_lr, cfg.loss.rate_quad)

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

    # ---- losses ----
    def _losses(self, batch: list[TrainItem], ratio: float, recon_w: float) -> dict[str, Tensor]:
        cfg, lm = self.cfg, self.lm
        pad_id = lm.tokenizer.pad_token_id
        ids, mask = pad_ids([it.ctx_ids for it in batch], pad_id, self.device)
        h = lm.token_features(ids, mask, cfg.llm.feature_layer)
        with torch.autocast(self.device.type, dtype=amp_dtype(), enabled=self.use_amp):
            mem = self.compressor(h, mask, ratio)
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
                self.rate.update(ratio, float(frac.detach()))
        if "exp_edges" in aux:
            out["edge_l0"] = (aux["exp_edges"] / aux["n_tokens"]).mean()
        for key in ("active", "mean_degree", "edge_entropy", "edge_mass"):  # diagnostics only
            if key in aux:
                out[key] = aux[key].float().mean().detach()
        return out

    # ---- checkpoints ----
    def _save(self, name: str = "ckpt.pt") -> None:
        torch.save(
            {
                "compressor": self.compressor.state_dict(),
                "opt": self.opt.state_dict(),
                "scaler": self.scaler.state_dict(),
                "rate": self.rate.state_dict(),
                "step": self.step,
            },
            self.out / name,
        )

    def _resume(self) -> None:
        path = self.out / "ckpt.pt"
        if not path.exists():
            return
        state = torch.load(path, map_location=self.device)
        self.compressor.load_state_dict(state["compressor"])
        self.opt.load_state_dict(state["opt"])
        self.scaler.load_state_dict(state["scaler"])
        self.rate.load_state_dict(state["rate"])
        self.step = state["step"]
        print(f"resumed from step {self.step}", flush=True)

    def _log(self, record: dict) -> None:
        with (self.out / "train_log.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    # ---- loop ----
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
            for a in range(t.grad_accum):
                micro = self.step * t.grad_accum + a
                ratio = self._ratio(micro)
                losses = self._losses(self._batch(micro), ratio, w)
                self.scaler.scale(losses["total"] / t.grad_accum).backward()
                for k, v in losses.items():
                    acc[k] = acc.get(k, 0.0) + float(v.detach()) / t.grad_accum
            self.scaler.unscale_(self.opt)
            gnorm = torch.nn.utils.clip_grad_norm_(self.compressor.parameters(), 1.0)
            self.scaler.step(self.opt)
            self.scaler.update()
            self.opt.zero_grad(set_to_none=True)
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
        torch.save(self.compressor.state_dict(), final)
        return final
