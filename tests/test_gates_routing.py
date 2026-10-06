import torch

from gistgraph.model.gates import HardConcreteGate
from gistgraph.model.routing import RoutedWriter
from gistgraph.train.rate import RateController

D = 16


def _units(b=2, m=12, lens=(12, 9), seed=0):
    torch.manual_seed(seed)
    u = torch.randn(b, m, D)
    valid = torch.zeros(b, m, dtype=torch.bool)
    for i, n in enumerate(lens):
        valid[i, :n] = True
    return u, valid


# ---------------- hard-concrete gates ----------------


def test_gate_eval_is_deterministic_and_zero_on_invalid():
    gate = HardConcreteGate(D).eval()
    z = torch.randn(2, 5, D)
    valid = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    g1, _ = gate(z, valid)
    g2, _ = gate(z, valid)
    assert torch.equal(g1, g2) and torch.all(g1[~valid] == 0)


def test_gate_starts_mostly_open():
    gate = HardConcreteGate(D, init_log_alpha=2.0).eval()
    g, exp_open = gate(torch.randn(1, 20, D), torch.ones(1, 20, dtype=torch.bool))
    assert (g > 0).all() and exp_open.item() > 18


def test_gate_train_samples_hit_exact_zero_and_one():
    gate = HardConcreteGate(D, init_log_alpha=0.0).train()
    g, _ = gate(torch.zeros(1, 4000, D), torch.ones(1, 4000, dtype=torch.bool))
    assert (g == 0).any() and (g == 1).any() and ((g > 0) & (g < 1)).any()


def test_expected_open_matches_monte_carlo():
    torch.manual_seed(0)
    gate = HardConcreteGate(D, init_log_alpha=-0.5).train()
    z = torch.zeros(1, 20000, D)
    valid = torch.ones(1, 20000, dtype=torch.bool)
    g, exp_open = gate(z, valid)
    empirical = (g > 0).float().mean().item()
    assert abs(exp_open.item() / 20000 - empirical) < 0.015


def test_expected_open_has_gradient_to_gate_parameters():
    gate = HardConcreteGate(D)
    _, exp_open = gate(torch.randn(2, 6, D), torch.ones(2, 6, dtype=torch.bool))
    exp_open.sum().backward()
    assert gate.proj.bias.grad.abs().sum() > 0


# ---------------- routed writer ----------------


def _writer(top_k=2, k_max=10, noise=0.0):
    return RoutedWriter(D, k_max, top_k=top_k, layers=1, noise=noise)


def test_each_segment_writes_to_exactly_top_k_valid_nodes():
    u, valid = _units()
    w = _writer(top_k=3).eval()
    counts = torch.tensor([8, 5])
    z, zv, aux = w(u, valid, counts, torch.tensor([4.0, 4.0]))
    write = aux["write"]  # [B, M, K]
    nonzero = (write > 0).sum(-1)
    assert torch.all(nonzero[valid] == 3)
    assert torch.all(nonzero[~valid] == 0)  # padded segments write nothing
    assert torch.all(write[1, :, 5:] == 0)  # doc 1 has only 5 nodes
    assert torch.allclose(write.sum(-1)[valid], torch.ones(int(valid.sum())), atol=1e-5)


def test_router_handles_fewer_nodes_than_top_k():
    u, valid = _units(b=1, lens=(12,))
    w = _writer(top_k=3).eval()
    z, zv, aux = w(u, valid, torch.tensor([1]), torch.tensor([4.0]))
    assert z.shape[1] == 1 and torch.isfinite(z).all()
    assert torch.allclose(aux["write"][0, :, 0], torch.ones(12))


def test_writer_masks_unused_nodes_and_is_finite():
    u, valid = _units()
    w = _writer().eval()
    z, zv, aux = w(u, valid, torch.tensor([6, 3]), torch.tensor([2.0, 4.0]))
    assert zv.sum(1).tolist() == [6, 3] and torch.all(z[1, 3:] == 0)
    assert torch.isfinite(z).all() and torch.isfinite(aux["lb"])


def test_load_balance_is_higher_when_routing_collapses():
    u, valid = _units(b=1, m=24, lens=(24,))
    counts, ratio = torch.tensor([8]), torch.tensor([3.0])
    w = _writer(k_max=8, top_k=1).eval()
    with torch.no_grad():
        w.route.weight.zero_()  # all segments score nodes identically -> ties broken by node keys
        w.keys.copy_(torch.randn_like(w.keys) * 0.0)
        w.keys[0] += 5.0  # node 0 wins for every segment
        w.route.weight.copy_(torch.eye(D))
    _, _, collapsed = w(u.abs() + 1.0, valid, counts, ratio)
    balanced = _writer(k_max=8, top_k=1).eval()
    with torch.no_grad():
        balanced.route.weight.zero_()
    _, _, even = balanced(u, valid, counts, ratio)
    assert collapsed["lb"] > even["lb"]


def test_gradients_reach_router_values_and_keys():
    u, valid = _units()
    w = _writer().train()
    z, _, aux = w(u, valid, torch.tensor([6, 4]), torch.tensor([3.0, 3.0]))
    (z.sum() + aux["lb"]).backward()
    assert w.route.weight.grad.abs().sum() > 0
    assert w.value.weight.grad.abs().sum() > 0 and w.keys.grad.abs().sum() > 0


def test_centroids_lie_in_unit_interval_for_written_nodes():
    u, valid = _units()
    w = _writer().eval()
    _, _, aux = w(u, valid, torch.tensor([6, 4]), torch.tensor([3.0, 3.0]))
    c, mass = aux["centroid"], aux["mass"]
    assert torch.all(c[mass > 1e-6] >= 0) and torch.all(c[mass > 1e-6] <= 1.0)
    assert torch.all(c[mass <= 1e-6] == 2.0)


# ---------------- rate controller ----------------


def test_rate_multiplier_grows_when_too_many_nodes_are_open():
    rc = RateController(lr=0.5)
    rc.update(4.0, open_fraction=0.9)  # target is 0.25
    assert rc.multiplier(4.0) > 0
    rc2 = RateController(lr=0.5)
    rc2.update(4.0, open_fraction=0.1)
    assert rc2.multiplier(4.0) < 0


def test_rate_controller_converges_on_toy_problem():
    torch.manual_seed(0)
    theta = torch.tensor(2.0, requires_grad=True)  # open fraction = sigmoid(theta)
    opt = torch.optim.Adam([theta], lr=0.05)
    rc = RateController(lr=0.1, rho=10.0)
    target = 0.25
    for _ in range(3000):
        frac = torch.sigmoid(theta)
        loss = 0.5 * (frac - 0.9) ** 2 + rc.penalty(4.0, frac)  # task loss wants 0.9
        opt.zero_grad()
        loss.backward()
        opt.step()
        rc.update(4.0, float(frac.detach()))
    assert abs(float(torch.sigmoid(theta)) - target) < 0.03


def test_rate_penalty_is_lagrangian_plus_quadratic():
    rc = RateController(lr=0.1, rho=4.0)
    rc.lam[4.0] = 2.0
    frac = torch.tensor(0.5)  # g = 0.25
    assert rc.penalty(4.0, frac).item() == 2.0 * 0.25 + 0.5 * 4.0 * 0.25**2


def test_rate_controller_state_roundtrip_and_limits():
    rc = RateController(lr=100.0, limit=5.0)
    rc.update(2.0, 1.0)
    assert rc.multiplier(2.0) == 5.0  # clipped
    other = RateController()
    other.load_state_dict(rc.state_dict())
    assert other.multiplier(2.0) == 5.0
