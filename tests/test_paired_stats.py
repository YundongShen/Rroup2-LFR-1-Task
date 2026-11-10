"""Tests for scripts/paired_stats.py against known values."""
import math

import numpy as np
import pytest

from scripts.paired_stats import paired_t, t_sf_two_sided, wilcoxon_exact


@pytest.mark.parametrize("t,dof,p", [(2.776445, 4, 0.05), (4.604095, 4, 0.01), (2.570582, 5, 0.05),
                                     (1.959964, 10 ** 6, 0.05), (0.0, 4, 1.0)])
def test_t_critical_values(t, dof, p):
    assert t_sf_two_sided(t, dof) == pytest.approx(p, abs=2e-5)


def test_t_dof4_closed_form():
    for t in (0.3, 1.0, 2.0, 5.0):
        y = t * t / (4 + t * t)
        assert t_sf_two_sided(t, 4) == pytest.approx(1 - (1.5 * math.sqrt(y) - 0.5 * y ** 1.5), abs=1e-8)


def test_paired_t():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = a - np.array([0.5, 0.1, 0.3, -0.2, 0.4])
    m, sd, t, p = paired_t(a, b)
    d = a - b
    assert m == pytest.approx(d.mean()) and sd == pytest.approx(d.std(ddof=1))
    assert t == pytest.approx(d.mean() / (d.std(ddof=1) / math.sqrt(5)))
    assert p == pytest.approx(t_sf_two_sided(t, 4))


def test_wilcoxon_exact():
    assert wilcoxon_exact([1, 2, 3, 4, 5], [0, 0, 0, 0, 0]) == pytest.approx(2 / 32)      # all positive, n=5
    assert wilcoxon_exact([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == 1.0                        # all zero
    # d = +1, +2, -3, +4, +5: W- = 3; P(min(W+, W-) <= 3) = 2 * #{W+ <= 3} / 32 = 2 * 5 / 32
    assert wilcoxon_exact([1, 2, -3, 4, 5], [0] * 5) == pytest.approx(10 / 32)
