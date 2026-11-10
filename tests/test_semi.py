import pandas as pd
import torch
import torch.nn.functional as F

from src.data import CAPTION_COLS
from src.methods.semi import (Adapter, assign_pseudo_captions, build_bank, contrastive_loss,
                              pcap_loss, zscore_rows)


def _toy_b():
    caps = [["a", "b", "b", "c", "d"], ["e", "a", "f", "g", "h"], ["i", "j", "k", "l", "m"]]
    return pd.DataFrame({c: [r[k] for r in caps] for k, c in enumerate(CAPTION_COLS)})


def test_build_bank_dedup_and_owners():
    bank, bank_id, owns = build_bank(_toy_b())
    assert bank == sorted(set("abcdefghijklm"))
    assert bank_id.shape == (3, 5)
    a = bank.index("a")
    assert owns[0, a] and owns[1, a] and not owns[2, a]          # "a" belongs to images 0 and 1
    assert owns.sum(1).tolist() == [4, 5, 5]                     # duplicates counted once


def test_adapter_identity_at_init():
    ad = Adapter(8, 16, 0.2)
    z = torch.randn(5, 8)
    assert torch.equal(ad(z), z)


def test_s_ii_matches_bruteforce_and_weights():
    torch.manual_seed(0)
    bank, bank_id, owns = build_bank(_toy_b())
    NA, d = 7, 6
    zA = F.normalize(torch.randn(NA, d), dim=-1)
    zB = F.normalize(torch.randn(3, d), dim=-1)
    tB = F.normalize(torch.randn(len(bank), d), dim=-1)
    cand, w, conf, keep = assign_pseudo_captions(zA, zB, tB, bank_id, 0.2, 3, 0.1, 0.5)
    S_img = zA @ zB.T
    S_ii = torch.stack([S_img[:, owns[:, j]].max(1).values for j in range(len(bank))], 1)
    R = 0.2 * zscore_rows(zA @ tB.T) + 0.8 * zscore_rows(S_ii)
    top, idx = R.topk(3, 1)
    assert torch.equal(cand, idx)
    assert torch.allclose(w, torch.softmax(top / 0.1, 1))
    assert torch.allclose(w.sum(1), torch.ones(NA))
    assert len(keep) == round(0.5 * NA)
    assert conf[keep].min() >= conf[[i for i in range(NA) if i not in keep.tolist()]].max()


def test_contrastive_masks_shared_captions_only():
    torch.manual_seed(0)
    zi = F.normalize(torch.randn(3, 4), dim=-1)
    tj = F.normalize(torch.randn(3, 4), dim=-1)
    owns = torch.eye(3, dtype=torch.bool)
    base = contrastive_loss(zi, tj, owns, 10.0)
    # caption 1 also belongs to image 0 -> removed from image 0's and caption 1's negatives
    owns2 = owns.clone()
    owns2[0, 1] = True
    masked = contrastive_loss(zi, tj, owns2, 10.0)
    logits = 10.0 * zi @ tj.T
    logits[0, 1] = -float("inf")
    y = torch.arange(3)
    ref = 0.5 * (F.cross_entropy(logits, y) + F.cross_entropy(logits.T, y))
    assert torch.allclose(masked, ref) and not torch.allclose(masked, base)


def test_pcap_loss_is_soft_ce_over_bank():
    torch.manual_seed(0)
    zu = F.normalize(torch.randn(2, 4), dim=-1)
    tB = F.normalize(torch.randn(5, 4), dim=-1)
    cand = torch.tensor([[0, 2, 4], [1, 3, 0]])
    w = torch.tensor([[0.5, 0.3, 0.2], [1.0, 0.0, 0.0]])
    logp = torch.log_softmax(10.0 * zu @ tB.T, 1)
    ref = -((0.5 * logp[0, 0] + 0.3 * logp[0, 2] + 0.2 * logp[0, 4]) + logp[1, 1]) / 2
    assert torch.allclose(pcap_loss(zu, tB, cand, w, 10.0), ref)
