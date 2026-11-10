"""Tests for the balanced caption->class bridge (src/methods/semi.py, TASK.md §5)."""
import pytest
import torch
import torch.nn.functional as F

from src.methods.semi import (bridge_loss, bridge_r, caption_class_matrix, class_prior, entropy_rows,
                              unbalanced_sinkhorn)


def _skewed_r(N=600, C=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(N, C, generator=g, dtype=torch.float64)
    logits[:, 0] += 1.5                                   # class 0 over-represented in r
    logits[:, 1] -= 1.5                                   # class 1 under-represented
    return torch.softmax(logits, 1)


@pytest.mark.parametrize("eps", [0.5, 1.0])
def test_kappa0_is_row_normalised_power(eps):
    r = _skewed_r()
    Q = unbalanced_sinkhorn(r, eps, torch.full((8,), 1 / 8, dtype=torch.float64), kappa=0.0)
    K = r ** (1 / eps)
    assert torch.allclose(Q, K / K.sum(1, keepdim=True), atol=1e-12)


@pytest.mark.parametrize("prior", ["uniform", "skewed"])
@pytest.mark.parametrize("eps", [0.5, 1.0])
def test_kappa1_column_sums_equal_N_prior(prior, eps):
    r = _skewed_r()
    N, C = r.shape
    pi = (torch.full((C,), 1 / C) if prior == "uniform" else torch.linspace(1, 2, C) / torch.linspace(1, 2, C).sum())
    Q = unbalanced_sinkhorn(r, eps, pi.double(), kappa=1.0, n_iter=5000)
    assert torch.allclose(Q.sum(1), torch.ones(N, dtype=Q.dtype), atol=1e-12)     # rows exactly 1
    assert torch.allclose(Q.sum(0), N * pi.double(), rtol=1e-6)


def test_kappa_half_column_sums_between():
    r = _skewed_r()
    N, C = r.shape
    pi = torch.full((C,), 1 / C, dtype=torch.float64)
    col = {k: unbalanced_sinkhorn(r, 1.0, pi, kappa=k, n_iter=5000).sum(0) for k in (0.0, 0.5, 1.0)}
    lo, hi = torch.minimum(col[0.0], col[1.0]), torch.maximum(col[0.0], col[1.0])
    assert ((col[0.5] >= lo - 1e-6) & (col[0.5] <= hi + 1e-6)).all()
    tv = {k: 0.5 * (v / N - pi).abs().sum() for k, v in col.items()}
    assert tv[1.0] < 1e-6 < tv[0.5] < tv[0.0]
    Q = unbalanced_sinkhorn(r, 1.0, pi, kappa=0.5, n_iter=5000)
    assert torch.allclose(Q.sum(1), torch.ones(N, dtype=Q.dtype), atol=1e-12)


def test_kappa_is_not_cumulative():
    """A cumulative Q *= (pi / colsum)^kappa update would reach full balance for any kappa > 0."""
    r = _skewed_r()
    pi = torch.full((8,), 1 / 8, dtype=torch.float64)
    Q1 = unbalanced_sinkhorn(r, 1.0, pi, kappa=0.5, n_iter=200)
    Q2 = unbalanced_sinkhorn(r, 1.0, pi, kappa=0.5, n_iter=5000)
    assert torch.allclose(Q1, Q2, atol=1e-8)                                      # converged fixed point
    assert 0.5 * (Q2.sum(0) / len(r) - pi).abs().sum() > 1e-3                      # still not balanced


def test_sinkhorn_float32_input_keeps_dtype():
    r = _skewed_r().float()
    Q = unbalanced_sinkhorn(r, 1.0, torch.full((8,), 1 / 8), kappa=1.0)
    assert Q.dtype == torch.float32 and torch.allclose(Q.sum(1), torch.ones(len(r)), atol=1e-5)


def test_caption_class_matrix_and_prior():
    g = torch.Generator().manual_seed(1)
    t = F.normalize(torch.randn(6, 5, generator=g), dim=-1)
    V = F.normalize(torch.randn(3, 5, generator=g), dim=-1)
    A = caption_class_matrix(t, V, 0.01)
    assert torch.allclose(A, torch.softmax(t @ V.T / 0.01, 1)) and torch.allclose(A.sum(1), torch.ones(6))
    slots = torch.tensor([[0, 0, 1, 2, 3], [4, 5, 5, 5, 1]])                   # duplicates count as they occur
    cap = A[slots.reshape(-1)].mean(0)
    assert torch.allclose(class_prior("caption", 0.3, A, slots), cap)
    assert torch.allclose(class_prior("uniform", 0.3, A, slots), torch.full((3,), 1 / 3))
    assert torch.allclose(class_prior("mix", 0.3, A, slots), 0.7 / 3 + 0.3 * cap)
    with pytest.raises(ValueError):
        class_prior("labels", 0.3, A, slots)


def test_bridge_r_matches_loop():
    g = torch.Generator().manual_seed(2)
    A = torch.softmax(torch.randn(10, 4, generator=g), 1)
    cand = torch.tensor([[1, 4, 7], [0, 0, 9], [3, 2, 1]])
    w = torch.softmax(torch.randn(3, 3, generator=g), 1)
    r = bridge_r(cand, w, A)
    ref = torch.stack([sum(w[u, k] * A[cand[u, k]] for k in range(3)) for u in range(3)])
    assert torch.allclose(r, ref) and torch.allclose(r.sum(1), torch.ones(3))


def test_bridge_loss_hard_and_soft():
    g = torch.Generator().manual_seed(3)
    zu = F.normalize(torch.randn(4, 6, generator=g), dim=-1)
    V = F.normalize(torch.randn(3, 6, generator=g), dim=-1)
    logp = torch.log_softmax(10 * zu @ V.T, 1)
    yhat = torch.tensor([0, 2, 1, 2])
    assert torch.allclose(bridge_loss(zu, V, 10.0, yhat=yhat), -logp[torch.arange(4), yhat].mean())
    Q = torch.softmax(torch.randn(4, 3, generator=g), 1)
    assert torch.allclose(bridge_loss(zu, V, 10.0, Q=Q), -(Q * logp).sum(1).mean())
    assert torch.allclose(bridge_loss(zu, V, 10.0, Q=F.one_hot(yhat, 3).float()),
                          bridge_loss(zu, V, 10.0, yhat=yhat))


def test_entropy_rows():
    p = torch.tensor([[1.0, 0.0], [0.5, 0.5]])
    assert torch.allclose(entropy_rows(p), torch.tensor([0.0, 0.6931472]), atol=1e-6)
