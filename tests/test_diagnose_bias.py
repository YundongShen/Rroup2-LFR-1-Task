"""Tests for the numpy analysis functions of scripts/diagnose_bias.py and its consistency with the repo."""
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

from scripts import diagnose_bias as db


# ----------------------------------------------------------------------------- consistency with the repository
def test_split_and_prompt_match_repo_configs():
    from src.data import b_folds
    cfg = OmegaConf.load("configs/data/kfold5.yaml")
    ns = SimpleNamespace(mode="kfold", k=5, stratify=True, seed=0)      # as hard-coded in diagnose_bias.main
    assert (ns.mode, ns.k, ns.stratify, ns.seed) == (cfg.b_split.mode, cfg.b_split.k,
                                                     cfg.b_split.stratify, cfg.b_split.seed)
    a, b = b_folds(cfg.manifest_dir, ns), b_folds(cfg.manifest_dir, cfg.b_split)
    assert all(x[1].filepath.tolist() == y[1].filepath.tolist() for x, y in zip(a, b))
    assert db.TEMPLATES == list(OmegaConf.load("configs/prompts/course.yaml").templates)


# ----------------------------------------------------------------------------- softmax helpers
def test_softmax_and_log_softmax():
    x = np.random.default_rng(0).normal(size=(4, 7)) * 50
    p = db.softmax(x, 1)
    assert np.allclose(p.sum(1), 1)
    assert np.allclose(np.exp(db.log_softmax(x, 1)), p)


# ----------------------------------------------------------------------------- Sinkhorn
def _biased_scores(N=620, C=31, bias=0.4, seed=0):
    rng = np.random.default_rng(seed)
    y = np.arange(N) % C
    S = 0.3 * np.eye(C)[y] + 0.05 * rng.normal(size=(N, C))
    S[:, 0] += bias                                  # injected class bias: class 0 dominates argmax
    return S, y


def test_sinkhorn_row_and_column_sums_uniform():
    S, _ = _biased_scores()
    N, C = S.shape
    Q = db.sinkhorn(S, eps=0.02)                     # setting used by the diagnosis (50 iterations)
    assert np.allclose(Q.sum(1), 1, atol=1e-8)
    assert np.allclose(Q.sum(0), N / C, rtol=2e-2)
    Q = db.sinkhorn(S, eps=0.1, n_iter=2000)           # converged
    assert np.allclose(Q.sum(1), 1, atol=1e-8)
    assert np.allclose(Q.sum(0), N / C, rtol=1e-6)


def test_sinkhorn_column_sums_follow_prior():
    S, _ = _biased_scores()
    N, C = S.shape
    prior = np.linspace(1, 3, C)
    Q = db.sinkhorn(S, eps=0.05, prior=prior, n_iter=1000)
    assert np.allclose(Q.sum(1), 1, atol=1e-8)
    assert np.allclose(Q.sum(0), N * prior / prior.sum(), rtol=1e-3)


def test_sinkhorn_recovers_injected_bias():
    S, y = _biased_scores()
    acc_raw = (S.argmax(1) == y).mean()
    acc_bal = (db.sinkhorn(S, eps=0.02).argmax(1) == y).mean()
    S_clean = S.copy()
    S_clean[:, 0] -= 0.4
    acc_clean = (S_clean.argmax(1) == y).mean()
    assert acc_raw < 0.2                              # bias 0.4 > signal 0.3: almost everything -> class 0
    assert acc_bal > 0.95 and acc_bal >= acc_clean - 0.02


# ----------------------------------------------------------------------------- reports
def test_bias_report_hand_example():
    y = np.array([0, 0, 1, 1, 2, 2])
    pred = np.array([0, 0, 0, 0, 1, 1])
    rep, share, recall = db.bias_report(pred, y, 3)
    assert rep["acc"] == pytest.approx(2 / 6)
    assert rep["mean_per_class_acc"] == pytest.approx(1 / 3)
    assert rep["tv_pred_vs_true"] == pytest.approx(1 / 3)
    assert rep["n_never_predicted"] == 1 and rep["n_under_predicted(<0.5x)"] == 1
    assert rep["n_over_predicted(>2x)"] == 0
    assert np.allclose(share, [4 / 6, 2 / 6, 0]) and np.allclose(recall, [1, 0, 0])


def test_bias_report_over_predicted_error_share():
    y = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    pred = np.array([0, 0, 0, 3, 0, 1, 2, 3])          # class 0 predicted 4/8 vs true 2/8 -> not > 2x
    rep, _, _ = db.bias_report(pred, y, 4)
    assert rep["n_over_predicted(>2x)"] == 0
    pred = np.array([0, 0, 0, 0, 0, 1, 2, 3])          # 5/8 > 2 * 2/8
    rep, _, _ = db.bias_report(pred, y, 4)
    assert rep["n_over_predicted(>2x)"] == 1 and rep["err_share_into_over_predicted"] == 1.0


def test_synonym_report():
    K = np.full((4, 4), 0.1)
    np.fill_diagonal(K, 1.0)
    K[0, 1] = K[1, 0] = 0.9
    K[2, 3] = K[3, 2] = 0.8
    y = np.array([0, 0, 0, 1, 2, 3])
    pred = np.array([1, 1, 2, 1, 2, 3])
    rep = db.synonym_report(pred, y, K, k=1)
    assert rep["err_share_into_1_text_nearest"] == pytest.approx(2 / 3)
    assert rep["chance_level"] == pytest.approx(1 / 3)


def test_caption_report():
    y = np.array([0, 1, 2, 0])
    pred = np.array([0, 2, 2, 1])                      # wrong on items 1 and 3
    r = np.array([[.8, .1, .1], [.1, .8, .1], [.1, .1, .8], [.1, .8, .1]])
    logp = np.log(np.eye(3)[pred] * 0.6 + 0.2 * (1 - np.eye(3)[pred]))
    rep = db.caption_report(pred, y, logp, r)
    assert rep["acc_caption_only"] == pytest.approx(3 / 4)
    assert rep["caption_right_when_image_wrong"] == pytest.approx(1 / 2)
    assert rep["acc_image_x_caption"] == pytest.approx(3 / 4)


def test_zscore_rows():
    x = np.random.default_rng(1).normal(size=(3, 50)) * 4 + 2
    z = db.zscore(x)
    assert np.allclose(z.mean(1), 0, atol=1e-7) and np.allclose(z.std(1), 1, atol=1e-6)


# ----------------------------------------------------------------------------- pseudo-captions vs training code
def test_pseudo_caption_r_matches_semi_assignment():
    torch = pytest.importorskip("torch")
    from src.methods.semi import assign_pseudo_captions
    rng = np.random.default_rng(0)
    Nv, Nt, M, d = 9, 12, 40, 16
    unit = lambda a: a / np.linalg.norm(a, axis=1, keepdims=True)
    z_val, z_tr, T_cap = unit(rng.normal(size=(Nv, d))), unit(rng.normal(size=(Nt, d))), unit(rng.normal(size=(M, d)))
    cap_ids = rng.integers(0, M, size=(Nt, 5))
    cap_ids[:, 1] = cap_ids[:, 0]                     # duplicated caption inside an image
    cap_ids[3, 2] = cap_ids[7, 4]                     # caption shared by two images
    r = db.pseudo_caption_r(z_val, z_tr, cap_ids, T_cap, np.eye(M))   # A_cap = identity -> r = weights on captions

    bank = np.unique(cap_ids)
    pos = {c: i for i, c in enumerate(bank)}
    bank_id = torch.tensor([[pos[c] for c in row] for row in cap_ids])
    f = lambda a: torch.tensor(a, dtype=torch.float64)
    cand, w, _, _ = assign_pseudo_captions(f(z_val), f(z_tr), f(T_cap[bank]), bank_id, 0.2, 3, 0.1, 1.0)
    r2 = np.zeros((Nv, M))
    np.put_along_axis(r2, bank[cand.numpy()], w.numpy(), 1)
    assert np.allclose(r, r2, atol=1e-6)
