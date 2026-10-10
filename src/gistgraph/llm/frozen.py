"""Wrapper around a frozen causal LM that can read soft prompts (embeddings) as context."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from gistgraph.data.prompts import split_prompt

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def pick_dtype(name: str) -> torch.dtype:
    """Resolve a dtype name. ``auto`` is bf16 where supported, fp16 on older GPUs, else fp32."""
    if name != "auto":
        if name not in DTYPES:
            raise ValueError(
                f"unsupported dtype {name!r}; expected auto or one of {sorted(DTYPES)}"
            )
        return DTYPES[name]
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if native_bf16() else torch.float16


def native_bf16() -> bool:
    """bf16 in hardware (Ampere or newer). Older GPUs such as the T4 only emulate it, slowly."""
    if not torch.cuda.is_available():
        return False
    try:
        return torch.cuda.is_bf16_supported(including_emulation=False)
    except TypeError:  # torch without the keyword
        return torch.cuda.get_device_capability()[0] >= 8


def pad_left(seqs: list[Tensor]) -> tuple[Tensor, Tensor]:
    """Left-pad a list of ``[L_i, d]`` embedding sequences. Returns ``[B, L, d]`` and a mask."""
    max_len = max(s.shape[0] for s in seqs)
    d = seqs[0].shape[1]
    out = seqs[0].new_zeros(len(seqs), max_len, d)
    mask = torch.zeros(len(seqs), max_len, dtype=torch.long, device=seqs[0].device)
    for i, s in enumerate(seqs):
        out[i, max_len - s.shape[0] :] = s
        mask[i, max_len - s.shape[0] :] = 1
    return out, mask


class FrozenLM(nn.Module):
    """A frozen decoder-only LM plus its tokenizer.

    Parameters never receive gradients, but gradients still flow *through* the model to any
    embeddings passed in, which is how the compressor is trained.
    """

    def __init__(self, model: nn.Module, tokenizer, grad_ckpt: bool = False):
        super().__init__()
        self.model = model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if grad_ckpt:
            self.model.gradient_checkpointing_enable()
            self.model.config.use_cache = False
        self.grad_ckpt = grad_ckpt

    @classmethod
    def from_pretrained(
        cls,
        name: str,
        dtype: str = "auto",
        grad_ckpt: bool = False,
        device=None,
        revision: str | None = None,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch_dtype = pick_dtype(dtype)
        try:
            model = AutoModelForCausalLM.from_pretrained(name, dtype=torch_dtype, revision=revision)
        except TypeError:  # older transformers only know ``torch_dtype``
            model = AutoModelForCausalLM.from_pretrained(
                name, torch_dtype=torch_dtype, revision=revision
            )
        tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
        obj = cls(model, tokenizer, grad_ckpt=grad_ckpt)
        return obj.to(device) if device is not None else obj

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def d_model(self) -> int:
        return self.model.get_input_embeddings().weight.shape[1]

    def train(self, mode: bool = True):
        # The LM stays in eval mode (no dropout) even when the compressor is being trained.
        return super().train(False)

    def embed(self, ids: Tensor) -> Tensor:
        return self.model.get_input_embeddings()(ids)

    @torch.no_grad()
    def token_features(self, ids: Tensor, mask: Tensor, layer: int) -> Tensor:
        """Hidden states of ``layer`` for each token: the compressor's input features."""
        out = self.model.model(
            input_ids=ids, attention_mask=mask, output_hidden_states=True, use_cache=False
        )
        return out.hidden_states[layer]

    def forward_embeds(self, embeds: Tensor, mask: Tensor) -> Tensor:
        # HF applies gradient checkpointing only in train mode. Qwen-class models have zero
        # dropout, so train mode does not change the outputs.
        self.model.train(self.grad_ckpt and torch.is_grad_enabled())
        try:
            return self.model(inputs_embeds=embeds, attention_mask=mask, use_cache=False).logits
        finally:
            self.model.eval()

    def _hidden(self, embeds: Tensor, mask: Tensor) -> Tensor:
        self.model.train(self.grad_ckpt and torch.is_grad_enabled())
        try:
            return self.model.model(inputs_embeds=embeds, attention_mask=mask, use_cache=False)[0]
        finally:
            self.model.eval()

    def build_answer_inputs(
        self,
        questions: list[str],
        prefixes: list[Tensor],
        answers: list[Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Right-padded ``pre + soft prefix + post + answer`` embeddings for teacher forcing.

        Returns ``embeds [B, L, d]``, ``mask [B, L]`` and ``starts [B]``, the index where each
        answer begins. Token ``t`` of the answer is predicted by position ``starts + t - 1``.
        """
        seqs, starts = [], []
        for q, prefix, ans in zip(questions, prefixes, answers, strict=True):
            pre, post = split_prompt(self.tokenizer, q)
            dtype = self.model.get_input_embeddings().weight.dtype
            parts = [self.embed(self.encode_text(pre)), prefix.to(dtype)]
            parts.append(self.embed(self.encode_text(post)))
            starts.append(sum(p.shape[0] for p in parts))
            parts.append(self.embed(ans))
            seqs.append(torch.cat(parts, dim=0))
        max_len = max(s.shape[0] for s in seqs)
        embeds = seqs[0].new_zeros(len(seqs), max_len, seqs[0].shape[1])
        mask = torch.zeros(len(seqs), max_len, dtype=torch.long, device=embeds.device)
        for i, s in enumerate(seqs):
            embeds[i, : s.shape[0]] = s
            mask[i, : s.shape[0]] = 1
        return embeds, mask, torch.tensor(starts, device=embeds.device)

    def answer_logits(
        self, embeds: Tensor, mask: Tensor, starts: Tensor, lens: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Logits ``[B, T, V]`` at the answer positions only, plus a ``[B, T]`` validity mask.

        Applying the LM head only where it is needed avoids a ``[B, L, V]`` logits tensor, which
        would be huge for a 150k-token vocabulary.
        """
        hidden = self._hidden(embeds, mask)
        t = int(lens.max())
        steps = torch.arange(t, device=hidden.device)[None, :]
        valid = steps < lens[:, None]
        pos = (starts[:, None] - 1 + steps).clamp(max=hidden.shape[1] - 1)
        picked = hidden.gather(1, pos.unsqueeze(-1).expand(-1, -1, hidden.shape[-1]))
        return self.model.get_output_embeddings()(picked), valid

    @torch.no_grad()
    def generate_scored(
        self, embeds: Tensor, mask: Tensor, max_new_tokens: int, topk: int
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Greedy generation that also returns the top-k logits at every step.

        Returns ``ids [B, T]``, ``lens [B]`` (tokens up to and including the first end token),
        ``topk_idx [B, T, k]`` and ``topk_logits [B, T, k]``.
        """
        self.model.eval()
        out = self.model.generate(
            inputs_embeds=embeds,
            attention_mask=mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.0,
            use_cache=True,
            pad_token_id=self.tokenizer.pad_token_id,
            output_scores=True,
            return_dict_in_generate=True,
        )
        ids = out.sequences
        scores = torch.stack(out.scores, dim=1)  # [B, T, V]
        top = scores.float().topk(topk, dim=-1)
        eos = self.model.generation_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else list(eos or [self.tokenizer.eos_token_id])
        is_end = torch.zeros_like(ids, dtype=torch.bool)
        for e in eos:
            is_end |= ids == e
        first_end = torch.where(
            is_end.any(1), is_end.float().argmax(1), torch.full_like(ids[:, 0], ids.shape[1] - 1)
        )
        return ids, first_end + 1, top.indices, top.values

    def encode_text(self, text: str) -> Tensor:
        ids = self.tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids
        return ids[0].to(self.device)

    def build_inputs(
        self, questions: list[str], prefixes: list[Tensor | None], contexts: list[str] | None = None
    ) -> tuple[Tensor, Tensor]:
        """Assemble ``pre + context + post`` embeddings for a batch (left-padded).

        Each context slot is filled either by ``prefixes[i]`` (soft embeddings ``[K, d]``) or,
        when that is ``None``, by the text ``contexts[i]``.
        """
        seqs = []
        for i, q in enumerate(questions):
            pre, post = split_prompt(self.tokenizer, q)
            parts = [self.embed(self.encode_text(pre))]
            if prefixes[i] is not None:
                parts.append(prefixes[i].to(parts[0].dtype))
            elif contexts is not None and contexts[i]:
                parts.append(self.embed(self.encode_text(contexts[i])))
            parts.append(self.embed(self.encode_text(post)))
            seqs.append(torch.cat(parts, dim=0))
        return pad_left(seqs)

    @torch.no_grad()
    def generate(self, embeds: Tensor, mask: Tensor, max_new_tokens: int = 32) -> list[str]:
        self.model.eval()
        out = self.model.generate(
            inputs_embeds=embeds,
            attention_mask=mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.0,  # override the model's default so decoding is plain greedy
            use_cache=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        return [t.strip() for t in self.tokenizer.batch_decode(out, skip_special_tokens=True)]
