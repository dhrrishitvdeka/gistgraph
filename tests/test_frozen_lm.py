import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from gistgraph.llm.frozen import FrozenLM, pad_left


class CharTokenizer:
    """Character-level tokenizer with the few attributes FrozenLM needs."""

    eos_token = "<eos>"
    pad_token = "<eos>"
    pad_token_id = 0

    def _ids(self, text):
        return [(ord(c) % 90) + 5 for c in text]

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        ids = torch.tensor([self._ids(text)])
        return type("Enc", (), {"input_ids": ids})()

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return f"<u>{messages[0]['content']}</u><a>"

    def batch_decode(self, out, skip_special_tokens=True):
        return ["".join(chr(int(i) + 30) for i in row) for row in out]


@pytest.fixture(scope="module")
def lm():
    torch.manual_seed(0)
    cfg = Qwen2Config(
        vocab_size=100,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=512,
    )
    return FrozenLM(Qwen2ForCausalLM(cfg), CharTokenizer())


def test_parameters_are_frozen(lm):
    assert all(not p.requires_grad for p in lm.model.parameters())
    assert lm.d_model == 32


def test_pad_left():
    a, b = torch.ones(2, 4), torch.ones(3, 4) * 2
    out, mask = pad_left([a, b])
    assert out.shape == (2, 3, 4)
    assert mask.tolist() == [[0, 1, 1], [1, 1, 1]]
    assert torch.equal(out[0, 1:], a) and torch.equal(out[0, 0], torch.zeros(4))


def test_build_inputs_text_matches_manual_concatenation(lm):
    q = "why?"
    embeds, mask = lm.build_inputs([q], [None], contexts=["hello"])
    from gistgraph.data.prompts import full_prompt

    ids = lm.encode_text(full_prompt(lm.tokenizer, "hello", q))
    assert torch.allclose(embeds[0], lm.embed(ids))
    assert mask.sum().item() == len(ids)


def test_build_inputs_soft_prefix_sits_between_pre_and_post(lm):
    prefix = torch.randn(3, lm.d_model)
    embeds, mask = lm.build_inputs(["why?", "who is it?"], [prefix, prefix])
    from gistgraph.data.prompts import split_prompt

    pre, _ = split_prompt(lm.tokenizer, "who is it?")
    n_pre = len(lm.encode_text(pre))
    row = embeds[1]  # longest question, so no left padding
    assert mask[1].all()
    assert torch.allclose(row[n_pre : n_pre + 3], prefix)
    assert mask[0].sum() < mask[1].sum()  # shorter question is left-padded


def test_gradient_flows_to_soft_prefix_but_not_into_weights(lm):
    prefix = torch.randn(4, lm.d_model, requires_grad=True)
    embeds, mask = lm.build_inputs(["why?"], [prefix])
    logits = lm.forward_embeds(embeds, mask)
    logits[:, -1].logsumexp(-1).sum().backward()
    assert prefix.grad is not None and prefix.grad.abs().sum() > 0
    assert all(p.grad is None for p in lm.model.parameters())


def test_token_features_shape(lm):
    ids = torch.randint(5, 90, (2, 7))
    feats = lm.token_features(ids, torch.ones_like(ids), layer=1)
    assert feats.shape == (2, 7, lm.d_model)
    assert not feats.requires_grad


def test_generate_returns_one_string_per_example(lm):
    embeds, mask = lm.build_inputs(["a?", "bb?"], [None, None], contexts=["x", "yy"])
    outs = lm.generate(embeds, mask, max_new_tokens=3)
    assert len(outs) == 2 and all(isinstance(o, str) for o in outs)


def test_checkpointed_forward_matches_plain(lm):
    prefix = torch.randn(3, lm.d_model, requires_grad=True)
    embeds, mask = lm.build_inputs(["why?"], [prefix])
    plain = lm.forward_embeds(embeds, mask)
    lm.model.gradient_checkpointing_enable()
    lm.grad_ckpt = True
    try:
        ckpt = lm.forward_embeds(embeds, mask)
    finally:
        lm.model.gradient_checkpointing_disable()
        lm.grad_ckpt = False
    assert torch.allclose(plain, ckpt, atol=1e-5)
