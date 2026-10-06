import pytest
import torch

from gistgraph.config import load_config
from gistgraph.model.compressor import FlatCompressor, build_compressor, max_slots
from gistgraph.model.encoder import SegmentEncoder, sinusoidal_positions
from gistgraph.model.projector import Projector, mean_embedding_norm
from gistgraph.model.writer import FlatSlotWriter, num_slots, slot_mask_from_counts

D_IN, D = 16, 32


def _cfg(**extra):
    over = ["encoder.d=32", "encoder.layers=1", "data.max_ctx_tokens=64", "train.ratios=[2,4]"]
    over = [f"compressor.{o}" if o.startswith("encoder") else o for o in over]
    over += [f"{k}={v}" for k, v in extra.items()]
    return load_config(overrides=over)


def _batch(b=2, n=24, lens=(24, 17)):
    torch.manual_seed(0)
    h = torch.randn(b, n, D_IN)
    mask = torch.zeros(b, n, dtype=torch.bool)
    for i, ln in enumerate(lens):
        mask[i, :ln] = True
    return h, mask


def test_sinusoidal_positions_shape_and_range():
    pe = sinusoidal_positions(10, 8)
    assert pe.shape == (10, 8) and pe.abs().max() <= 1.0
    assert not torch.allclose(pe[0], pe[1])


@pytest.mark.parametrize("seg_len", [1, 4])
def test_encoder_shapes(seg_len):
    h, mask = _batch()
    enc = SegmentEncoder(D_IN, D, layers=1, heads=4, seg_len=seg_len)
    u, valid = enc(h, mask)
    m = -(-24 // seg_len)
    assert u.shape == (2, m, D) and valid.shape == (2, m)
    assert valid[0].sum() == -(-24 // seg_len) and valid[1].sum() == -(-17 // seg_len)


def test_encoder_ignores_padded_content():
    h, mask = _batch()
    enc = SegmentEncoder(D_IN, D, layers=1, heads=4).eval()
    u1, valid = enc(h, mask)
    h2 = h.clone()
    h2[~mask] = 123.0  # garbage in the padding must not matter
    u2, _ = enc(h2, mask)
    assert torch.allclose(u1[valid], u2[valid], atol=1e-5)


def test_num_slots_and_mask():
    counts = num_slots(torch.tensor([100, 7, 1]), torch.tensor([4.0, 3.0, 5.0]), k_max=20)
    assert counts.tolist() == [20, 2, 1]  # 25 capped to 20, floor(7/3)=2, at least 1
    mask = slot_mask_from_counts(torch.tensor([2, 4]), 4)
    assert mask.tolist() == [[True, True, False, False], [True] * 4]


def test_writer_outputs_and_masking():
    h, mask = _batch()
    enc = SegmentEncoder(D_IN, D, layers=1, heads=4)
    u, uv = enc(h, mask)
    writer = FlatSlotWriter(D, k_max=12, layers=1, heads=4)
    counts = torch.tensor([6, 3])
    z, zv, _ = writer(u, uv, counts, torch.tensor([4.0, 4.0]))
    assert z.shape == (2, 6, D)
    assert zv.sum(1).tolist() == [6, 3]
    assert torch.all(z[1, 3:] == 0)  # unused slots are zeroed


def test_writer_valid_slots_do_not_depend_on_unused_slots():
    h, mask = _batch(b=1, lens=(24,))
    enc = SegmentEncoder(D_IN, D, layers=1, heads=4).eval()
    writer = FlatSlotWriter(D, k_max=12, layers=1, heads=4).eval()
    u, uv = enc(h, mask)
    r = torch.tensor([4.0])
    z_small, _, _ = writer(u, uv, torch.tensor([3]), r)
    with torch.no_grad():
        writer.queries[3:] += 5.0  # perturb slots that are masked out in the small run
    z_small2, _, _ = writer(u, uv, torch.tensor([3]), r)
    assert torch.allclose(z_small, z_small2, atol=1e-5)


def test_projector_matches_target_norm():
    norm = 0.7
    proj = Projector(D, 48, norm)
    out = proj(torch.randn(3, 5, D) * 10)
    assert torch.allclose(out.norm(dim=-1), torch.full((3, 5), norm), atol=1e-4)


def test_mean_embedding_norm():
    w = torch.tensor([[3.0, 4.0], [0.0, 1.0]])
    assert mean_embedding_norm(w) == pytest.approx(3.0)


def test_flat_compressor_end_to_end_and_gradients():
    cfg = _cfg()
    comp = build_compressor(cfg, D_IN, 0.5)
    assert isinstance(comp, FlatCompressor) and comp.k_max == max_slots(cfg) == 32
    h, mask = _batch()
    mem = comp(h, mask, 4.0)
    assert mem.counts.tolist() == [6, 4]  # floor(24/4), floor(17/4)
    assert mem.embeds.shape == (2, 6, D_IN) and mem.prefix(1).shape == (4, D_IN)
    mem.embeds.sum().backward()
    assert comp.writer.queries.grad.abs().sum() > 0
    assert all(p.grad is not None for p in comp.encoder.parameters())


def test_flat_compressor_different_ratios_give_different_slot_counts():
    comp = build_compressor(_cfg(), D_IN, 0.5)
    h, mask = _batch()
    assert comp(h, mask, 2.0).counts.tolist() == [12, 8]
    assert comp(h, mask, 5.0).counts.tolist() == [4, 3]


def test_unknown_compressor_type():
    with pytest.raises(KeyError):
        build_compressor(_cfg(**{"compressor.type": "nope"}), D_IN, 0.5)
