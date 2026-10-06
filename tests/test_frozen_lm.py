import torch

from gistgraph.llm.frozen import pad_left


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


def _answer_batch(lm):
    prefixes = [torch.randn(3, lm.d_model, requires_grad=True), torch.randn(5, lm.d_model)]
    answers = [torch.tensor([7, 8, 9]), torch.tensor([11, 12])]
    embeds, mask, starts = lm.build_answer_inputs(["why?", "who is it?"], prefixes, answers)
    return prefixes, answers, embeds, mask, starts


def test_build_answer_inputs_layout(lm):
    prefixes, answers, embeds, mask, starts = _answer_batch(lm)
    assert mask.sum(1).tolist() == [s + len(a) for s, a in zip(starts.tolist(), answers)]
    assert torch.allclose(embeds[0, starts[0] : starts[0] + 3], lm.embed(answers[0]))
    assert torch.allclose(embeds[1, starts[1] : starts[1] + 2], lm.embed(answers[1]))


def test_answer_logits_match_full_forward_at_answer_positions(lm):
    _, answers, embeds, mask, starts = _answer_batch(lm)
    lens = torch.tensor([3, 2])
    got, valid = lm.answer_logits(embeds, mask, starts, lens)
    full = lm.forward_embeds(embeds, mask)
    assert got.shape == (2, 3, full.shape[-1]) and valid.tolist() == [[1, 1, 1], [1, 1, 0]]
    for b, n in enumerate(lens.tolist()):
        for t in range(n):
            assert torch.allclose(got[b, t], full[b, starts[b] - 1 + t], atol=1e-5)


def test_answer_logits_pass_gradient_to_prefix(lm):
    prefixes, _, embeds, mask, starts = _answer_batch(lm)
    # embeds were built from the prefixes, so rebuild with grad tracking intact
    logits, valid = lm.answer_logits(embeds, mask, starts, torch.tensor([3, 2]))
    logits.sum().backward()
    assert prefixes[0].grad is not None and prefixes[0].grad.abs().sum() > 0


def test_generate_scored_shapes_and_lengths(lm):
    embeds, mask = lm.build_inputs(["a?", "bb?"], [None, None], contexts=["x", "yy"])
    ids, lens, idx, logits = lm.generate_scored(embeds, mask, max_new_tokens=4, topk=5)
    assert ids.shape[0] == 2 and idx.shape == (2, ids.shape[1], 5) == logits.shape
    assert ((lens >= 1) & (lens <= ids.shape[1])).all()
    # top-1 of the recorded logits is the greedy token that was emitted
    assert torch.equal(idx[:, :, 0], ids)


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
