"""Paired tests without scipy: two-sided paired t-test and exact Wilcoxon signed-rank test."""
import itertools
import math

import numpy as np


def t_sf_two_sided(t, dof, n_grid=20001):
    """P(|T| >= |t|) for Student's t with `dof` degrees of freedom (Simpson integration of the density)."""
    t = abs(float(t))
    if math.isinf(t):
        return 0.0
    c = math.exp(math.lgamma((dof + 1) / 2) - math.lgamma(dof / 2)) / math.sqrt(dof * math.pi)
    x = np.linspace(0.0, t, n_grid)
    f = c * (1 + x ** 2 / dof) ** (-(dof + 1) / 2)
    h = x[1] - x[0] if t > 0 else 0.0
    integral = h / 3 * (f[0] + f[-1] + 4 * f[1:-1:2].sum() + 2 * f[2:-1:2].sum())
    return float(max(0.0, 1.0 - 2.0 * integral))


def paired_t(a, b):
    """Two-sided paired t-test of a - b. Returns (mean diff, std diff (ddof=1), t, p)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    sd = d.std(ddof=1)
    if sd == 0:
        return float(d.mean()), 0.0, (math.inf if d.mean() != 0 else 0.0), (0.0 if d.mean() != 0 else 1.0)
    t = d.mean() / (sd / math.sqrt(n))
    return float(d.mean()), float(sd), float(t), t_sf_two_sided(t, n - 1)


def wilcoxon_exact(a, b):
    """Exact two-sided Wilcoxon signed-rank p-value (zeros dropped, average ranks for ties)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return 1.0
    order = np.argsort(np.abs(d))
    ranks = np.empty(n)
    absd = np.abs(d)[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and absd[j + 1] == absd[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    w_plus = ranks[d > 0].sum()
    total = ranks.sum()
    stat = min(w_plus, total - w_plus)
    count = sum(1 for signs in itertools.product((0, 1), repeat=n)
                if min(s := sum(r for r, g in zip(ranks, signs) if g), total - s) <= stat + 1e-12)
    return min(1.0, count / 2 ** n)
