"""Wrapper around a frozen causal LM that can read soft prompts (embeddings) as context."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from gistgraph.data.prompts import split_prompt


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
    def from_pretrained(cls, name: str, dtype: str = "auto", grad_ckpt: bool = False, device=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch_dtype = "auto" if dtype == "auto" else getattr(torch, dtype)
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch_dtype)
        tokenizer = AutoTokenizer.from_pretrained(name)
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
