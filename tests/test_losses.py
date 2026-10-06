import math

import pytest
import torch

from gistgraph.train.losses import ce_loss, kd_loss, lr_at, recon_weight


def _teacher_from(logits, k):
    top = logits.topk(k, dim=-1)
    return top.indices, top.values


def test_kd_is_zero_when_student_matches_teacher_on_full_support():
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 8)
    idx, vals = _teacher_from(logits, 8)  # k == vocab, so no tail is dropped
    valid = torch.ones(2, 3)
    assert kd_loss(logits, idx, vals, valid).item() == pytest.approx(0.0, abs=1e-6)


def test_kd_positive_and_has_gradient_when_student_differs():
    torch.manual_seed(0)
    teacher = torch.randn(2, 3, 8)
    student = torch.randn(2, 3, 8, requires_grad=True)
    idx, vals = _teacher_from(teacher, 4)
    loss = kd_loss(student, idx, vals, torch.ones(2, 3))
    loss.backward()
    assert loss.item() > 0 and student.grad.abs().sum() > 0


def test_kd_ignores_masked_positions():
    torch.manual_seed(1)
    teacher, student = torch.randn(1, 4, 8), torch.randn(1, 4, 8)
    idx, vals = _teacher_from(teacher, 4)
    valid = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    full = kd_loss(student, idx, vals, valid)
    corrupted = student.clone()
    corrupted[:, 2:] = 100.0
    assert kd_loss(corrupted, idx, vals, valid).item() == pytest.approx(full.item())


def test_kd_temperature_scaling_factor():
    torch.manual_seed(2)
    teacher, student = torch.randn(1, 2, 6), torch.randn(1, 2, 6)
    idx, vals = _teacher_from(teacher, 6)
    valid = torch.ones(1, 2)
    assert (
        kd_loss(student, idx, vals, valid, temp=2.0).item()
        != kd_loss(student, idx, vals, valid, temp=1.0).item()
    )


def test_ce_loss_matches_manual():
    logits = torch.tensor([[[2.0, 0.0], [0.0, 3.0]]])
    targets = torch.tensor([[0, 1]])
    expected = (math.log(1 + math.exp(-2.0)) + math.log(1 + math.exp(-3.0))) / 2
    assert ce_loss(logits, targets, torch.ones(1, 2)).item() == pytest.approx(expected)
    only_first = ce_loss(logits, targets, torch.tensor([[1.0, 0.0]]))
    assert only_first.item() == pytest.approx(math.log(1 + math.exp(-2.0)))


def test_lr_schedule_shape():
    assert lr_at(0, 100, 1.0, warmup=10) == pytest.approx(0.1)
    assert lr_at(9, 100, 1.0, warmup=10) == pytest.approx(1.0)
    assert lr_at(100, 100, 1.0, warmup=10) == pytest.approx(0.1)  # floor
    mid = lr_at(55, 100, 1.0, warmup=10)
    assert 0.1 < mid < 1.0


def test_recon_weight_anneals_to_zero():
    assert recon_weight(0, 1000, 1.0, 0.3) == 1.0
    assert recon_weight(150, 1000, 1.0, 0.3) == pytest.approx(0.5)
    assert recon_weight(300, 1000, 1.0, 0.3) == 0.0
    assert recon_weight(900, 1000, 1.0, 0.3) == 0.0
