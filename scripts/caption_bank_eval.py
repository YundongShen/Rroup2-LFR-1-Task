"""Offline check of dual-path caption retrieval, simulated on Split B.

Split B is divided once (seed-controlled) into query images, treated as if they had no
captions, and a caption bank built from the remaining B images and their captions.
Each query is scored against every bank caption by three rankings:

  IT      image->text cosine, the usual retrieval path
  II      image->image cosine to the caption's source image(s); a caption string that
          belongs to several bank images takes the most similar one
  fused   R = lam * z(IT) + (1 - lam) * z(II), z-scored per query over the whole bank
  two-stage@K   stage 1 keeps the K bank images nearest to the query (S_II); stage 2
          ranks the captions of those images by S_IT alone. A caption string enters the
          pool when any of its source images is among the K.

The query's own five captions are not in the bank, so quality is measured by the
frozen CLIP text-text cosine between a candidate and those five held-out captions
(mean over the five as the main score, max as the auxiliary one), averaged over the
top-1 / top-5 / top-10 candidates. A random-candidate floor and an oracle ceiling (the
single best bank caption per query) frame the numbers.

Also records, per query at the fixed export lam, the two rejection signals R_max (top
fused score) and agreement (overlap of the IT and II top-5 lists) and buckets quality by
both; for two-stage@K the signal is the top in-pool S_IT, bucketed the same way and also
read as an acceptance threshold (coverage vs. quality of the accepted queries).

Everything is computed from the cached frozen features; no image is re-encoded. The
scoring functions are shared with the trainer (src/experiment/pseudo_caption.py).

Writes to <out>/:
  summary.json           split sizes, chosen lam, duplicate / exact-match statistics
  sweep.csv / sweep.png  quality of top-1/5/10 against lam and against K, plus random and oracle
  rejection_rmax.csv, rejection_agreement.csv, rejection.png   quality by R_max bucket / agreement
  rejection_twostage.csv, threshold_twostage.csv, rejection_twostage.png
                         quality by top in-pool S_IT bucket, and coverage/quality per threshold
  index.csv              one row per query with its per-method scores and signals
  vis/<image_stem>/      a copy of the query image and candidates.json for manual review

    python scripts/caption_bank_eval.py
    python scripts/caption_bank_eval.py --seed 1 --query-frac 0.2 --vis-lam 0.2 --vis-k 5
    python scripts/caption_bank_eval.py --features data/features/vit_b16_laion2b.pt --n-vis 50
"""
import argparse
import json
import os
import shutil
import sys

import matplotlib
import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.data import CAPTION_COLS  # noqa: E402
from src.experiment.features import FeatureCache  # noqa: E402
from src.experiment.pseudo_caption import build_caption_bank, caption_image_scores, zscore  # noqa: E402

TOPKS = (1, 5, 10)
COLORS = {"top1": "#2a78d6", "top5": "#eb6834", "top10": "#1baf7a", "random": "#8a8a85", "oracle": "#eda100"}


def split_queries(n, query_frac, seed):
    rng = np.random.RandomState(seed)
    order = rng.permutation(n)
    n_query = int(round(n * query_frac))
    return np.sort(order[:n_query]), np.sort(order[n_query:])


def topk_quality(ranking, quality, ks=TOPKS, valid=None):
    """ranking [nq, n] candidate indices best first; quality [nq, n]. Per-query mean of the top-k.

    valid [nq, n] (bool) restricts the mean to candidates inside a pool; a pool smaller
    than k averages over what it has.
    """
    idx = ranking[:, :max(ks)]
    gathered = quality.gather(1, idx)
    mask = torch.ones_like(gathered) if valid is None else valid.gather(1, idx).float()
    return {k: (gathered[:, :k] * mask[:, :k]).sum(1) / mask[:, :k].sum(1).clamp_min(1) for k in ks}


def two_stage(s_img, s_it, sources, k):
    """Pool = captions of the k nearest bank images; ranked by S_IT inside the pool. Returns (ranking, pool mask)."""
    nearest = s_img.topk(k, dim=1).indices                                    # [nq, k]
    in_pool_img = torch.zeros_like(s_img, dtype=torch.bool).scatter_(1, nearest, True)
    cap_idx = torch.tensor([j for j, src in enumerate(sources) for _ in src])
    img_idx = torch.tensor([i for src in sources for i in src])
    pool = torch.zeros(s_img.shape[0], len(sources), dtype=torch.bool)
    pool.scatter_reduce_(1, cap_idx.expand(s_img.shape[0], -1), in_pool_img[:, img_idx], reduce="amax")
    ranking = s_it.double().masked_fill(~pool, float("-inf")).argsort(1, descending=True)
    return ranking, pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=os.path.join(ROOT, "data/features/vit_b32_laion2b.pt"))
    ap.add_argument("--manifest", default=os.path.join(ROOT, "data/splits/B_labeled.csv"))
    ap.add_argument("--out", default=None, help="default runs/caption_bank/seed<seed>")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--query-frac", type=float, default=0.2)
    ap.add_argument("--lams", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1")
    ap.add_argument("--vis-lam", type=float, default=0.2, help="fixed lam used for the manual-review export")
    ap.add_argument("--ks", default="1,2,3,5,10,20,50", help="K sweep for two-stage image-NN caption reranking")
    ap.add_argument("--vis-k", type=int, default=5, help="fixed K used for the manual-review export")
    ap.add_argument("--vis-top", type=int, default=5, help="candidates per method in candidates.json")
    ap.add_argument("--n-vis", type=int, default=None, help="export only the first N queries; default all")
    ap.add_argument("--rmax-bins", type=int, default=5)
    args = ap.parse_args()

    out = args.out or os.path.join(ROOT, "runs", "caption_bank", f"seed{args.seed}")
    os.makedirs(out, exist_ok=True)
    lams = [float(x) for x in args.lams.split(",")]
    if args.vis_lam not in lams:
        lams.append(args.vis_lam)
    lams = sorted(set(lams))
    ks = sorted(set(int(x) for x in args.ks.split(",")) | {args.vis_k})

    df = pd.read_csv(args.manifest)
    df["filepath"] = df["filepath"].astype(str)
    blob = torch.load(args.features, map_location="cpu", weights_only=False)
    cache = FeatureCache(blob)
    q_idx, b_idx = split_queries(len(df), args.query_frac, args.seed)
    query_df, bank_df = df.iloc[q_idx].reset_index(drop=True), df.iloc[b_idx].reset_index(drop=True)

    bank_caps, bank_src = build_caption_bank(bank_df, CAPTION_COLS)
    q_img = cache.images(query_df["filepath"].tolist())                       # [nq, d]
    b_img = cache.images(bank_df["filepath"].tolist())                        # [nb, d]
    b_txt = cache.texts(bank_caps)                                            # [nt, d]
    gt_caps = [[str(v) for v in row] for row in query_df[CAPTION_COLS].to_numpy()]
    gt_txt = torch.stack([cache.texts(caps) for caps in gt_caps])              # [nq, 5, d]
    nq, nt = q_img.shape[0], b_txt.shape[0]

    s_it = q_img @ b_txt.T                                                    # [nq, nt]  cosine, features are unit-norm
    s_img = q_img @ b_img.T                                                   # [nq, nb]
    s_ii = caption_image_scores(s_img, bank_src)                                 # [nq, nt]
    z_it, z_ii = zscore(s_it), zscore(s_ii)

    t2t = torch.einsum("qkd,td->qkt", gt_txt, b_txt)                          # [nq, 5, nt]
    q_mean, q_max = t2t.mean(1), t2t.amax(1)
    exact = torch.tensor([[c in set(g) for c in bank_caps] for g in gt_caps])  # [nq, nt]

    def rank(score):
        # the five captions of one bank image share one S_II; z(IT) breaks those ties, in float64
        return (score.double() + 1e-6 * z_it.double()).argsort(1, descending=True)

    fused = {lam: lam * z_it + (1 - lam) * z_ii for lam in lams}
    rank_it, rank_ii = rank(z_it), rank(z_ii)
    rankings = {"IT": rank_it, "II": rank_ii}
    rankings.update({f"fused@{lam:g}": rank(fused[lam]) for lam in lams})
    pools = {}
    for k in ks:
        rankings[f"twostage@K{k}"], pools[k] = two_stage(s_img, s_it, bank_src, k)
    valid = {name: pools[int(name.split("@K")[1])] for name in rankings if name.startswith("twostage@")}

    rng = np.random.RandomState(args.seed)
    rank_random = torch.tensor(np.stack([rng.permutation(nt)[:max(TOPKS)] for _ in range(nq)]))
    rankings["random"] = rank_random
    oracle_mean, oracle_max = q_mean.amax(1), q_max.amax(1)

    # sweep table ---------------------------------------------------------------------
    rows = []
    for name, rank in rankings.items():
        v = valid.get(name)
        tq_mean, tq_max = topk_quality(rank, q_mean, valid=v), topk_quality(rank, q_max, valid=v)
        exact_top5 = exact.gather(1, rank[:, :5]).float().mean(1)
        row = {"method": name, "lam": float(name.split("@")[1]) if name.startswith("fused@") else np.nan,
               "K": int(name.split("@K")[1]) if name.startswith("twostage@") else np.nan,
               "pool_size": v.sum(1).float().mean().item() if v is not None else np.nan}
        row.update({f"mean_cos@{k}": tq_mean[k].mean().item() for k in TOPKS})
        row.update({f"max_cos@{k}": tq_max[k].mean().item() for k in TOPKS})
        row["exact_match_rate@5"] = exact_top5.mean().item()
        rows.append(row)
    rows.append({"method": "oracle", "lam": np.nan, "mean_cos@1": oracle_mean.mean().item(),
                 "max_cos@1": oracle_max.mean().item()})
    sweep = pd.DataFrame(rows)
    sweep.to_csv(os.path.join(out, "sweep.csv"), index=False)

    # rejection signals at the fixed vis lam -----------------------------------------
    vis_rank = rankings[f"fused@{args.vis_lam:g}"]
    r_max = fused[args.vis_lam].amax(1)
    agreement = torch.tensor([len(set(rank_it[u, :5].tolist()) & set(rank_ii[u, :5].tolist())) for u in range(nq)])
    vis_q = topk_quality(vis_rank, q_mean)
    ts_name = f"twostage@K{args.vis_k}"
    ts_rank, ts_pool = rankings[ts_name], pools[args.vis_k]
    ts_q = topk_quality(ts_rank, q_mean, valid=ts_pool)
    ts_top_sit = s_it.masked_fill(~ts_pool, float("-inf")).amax(1)
    per_query = pd.DataFrame({
        "query_id": range(nq),
        "filepath": query_df["filepath"],
        "image": [os.path.splitext(os.path.basename(p))[0] for p in query_df["filepath"]],
        "r_max": r_max.numpy(),
        "agreement": agreement.numpy(),
        "twostage_top_s_it": ts_top_sit.numpy(),
        "oracle_mean_cos": oracle_mean.numpy(),
    })
    for name in ("IT", "II", f"fused@{args.vis_lam:g}", ts_name):
        tq = topk_quality(rankings[name], q_mean, valid=valid.get(name))
        for k in TOPKS:
            per_query[f"{name}_mean_cos@{k}"] = tq[k].numpy()
    per_query.to_csv(os.path.join(out, "index.csv"), index=False)

    quant = np.quantile(r_max.numpy(), np.linspace(0, 1, args.rmax_bins + 1))
    bucket = np.clip(np.searchsorted(quant, r_max.numpy(), side="right") - 1, 0, args.rmax_bins - 1)
    rej_r = pd.DataFrame({"bucket": bucket, "r_max": r_max.numpy(), "top1": vis_q[1].numpy(),
                          "top5": vis_q[5].numpy(), "top10": vis_q[10].numpy()})
    rej_r = rej_r.groupby("bucket").agg(n=("r_max", "size"), r_max_lo=("r_max", "min"), r_max_hi=("r_max", "max"),
                                        top1=("top1", "mean"), top5=("top5", "mean"), top10=("top10", "mean"))
    rej_r.to_csv(os.path.join(out, "rejection_rmax.csv"))
    rej_a = pd.DataFrame({"agreement": agreement.numpy(), "top1": vis_q[1].numpy(), "top5": vis_q[5].numpy(),
                          "top10": vis_q[10].numpy()})
    rej_a = rej_a.groupby("agreement").agg(n=("top1", "size"), top1=("top1", "mean"), top5=("top5", "mean"),
                                           top10=("top10", "mean"))
    rej_a.to_csv(os.path.join(out, "rejection_agreement.csv"))

    tq = np.quantile(ts_top_sit.numpy(), np.linspace(0, 1, args.rmax_bins + 1))
    tb = np.clip(np.searchsorted(tq, ts_top_sit.numpy(), side="right") - 1, 0, args.rmax_bins - 1)
    rej_t = pd.DataFrame({"bucket": tb, "top_s_it": ts_top_sit.numpy(), "top1": ts_q[1].numpy(),
                          "top5": ts_q[5].numpy(), "top10": ts_q[10].numpy()})
    rej_t = rej_t.groupby("bucket").agg(n=("top_s_it", "size"), s_it_lo=("top_s_it", "min"), s_it_hi=("top_s_it", "max"),
                                        top1=("top1", "mean"), top5=("top5", "mean"), top10=("top10", "mean"))
    rej_t.to_csv(os.path.join(out, "rejection_twostage.csv"))
    # acceptance threshold on the top in-pool S_IT: coverage and quality of what is accepted
    order = ts_top_sit.argsort(descending=True)
    thr_rows = []
    for n_acc in range(1, nq + 1):
        acc = order[:n_acc]
        thr_rows.append({"threshold": ts_top_sit[acc[-1]].item(), "coverage": n_acc / nq,
                         "top1": ts_q[1][acc].mean().item(), "top5": ts_q[5][acc].mean().item()})
    thr = pd.DataFrame(thr_rows)
    thr.to_csv(os.path.join(out, "threshold_twostage.csv"), index=False)

    # plots ---------------------------------------------------------------------------
    fused_rows = sweep[sweep["method"].str.startswith("fused@")].sort_values("lam")
    rnd = sweep[sweep["method"] == "random"].iloc[0]
    ts_rows = sweep[sweep["method"].str.startswith("twostage@")].sort_values("K")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, rows_, xcol, xlabel, title in (
            (axes[0], fused_rows, "lam", "λ   (0 = image→image only, 1 = image→text only)", "Fused ranking"),
            (axes[1], ts_rows, "K", "K nearest bank images (log scale)", "Two-stage: image NN, then S_IT")):
        for k in TOPKS:
            ax.plot(rows_[xcol], rows_[f"mean_cos@{k}"], marker="o", ms=4, lw=2, color=COLORS[f"top{k}"], label=f"top-{k}")
        ax.axhline(oracle_mean.mean().item(), color=COLORS["oracle"], lw=1.5, ls="--", label="oracle (best in bank)")
        ax.axhline(rnd["mean_cos@5"], color=COLORS["random"], lw=1.5, ls=":", label="random top-5")
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[1].set_xscale("log")
    axes[1].set_xticks(ts_rows["K"])
    axes[1].set_xticklabels([str(k) for k in ts_rows["K"]])
    axes[0].set_ylabel("mean text–text cosine to held-out captions")
    axes[0].legend(frameon=False)
    fig.suptitle(f"Caption retrieval on B, seed {args.seed}: {nq} queries, {nt} bank captions")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "sweep.png"), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(rej_r))
    for k in TOPKS:
        axes[0].plot(x, rej_r[f"top{k}"], marker="o", ms=4, lw=2, color=COLORS[f"top{k}"], label=f"top-{k}")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{lo:.2f}–{hi:.2f}\nn={n}" for lo, hi, n in zip(rej_r["r_max_lo"], rej_r["r_max_hi"], rej_r["n"])],
                            fontsize=8)
    axes[0].set_xlabel(f"R_max bucket (quantiles, λ={args.vis_lam:g})")
    axes[0].set_ylabel("mean text–text cosine")
    axes[0].set_title("Quality by R_max")
    for k in TOPKS:
        axes[1].plot(rej_a.index, rej_a[f"top{k}"], marker="o", ms=4, lw=2, color=COLORS[f"top{k}"], label=f"top-{k}")
    axes[1].set_xticks(rej_a.index)
    axes[1].set_xticklabels([f"{a}\nn={n}" for a, n in zip(rej_a.index, rej_a["n"])], fontsize=8)
    axes[1].set_xlabel("agreement = |top5(IT) ∩ top5(II)|")
    axes[1].set_title("Quality by agreement")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "rejection.png"), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(rej_t))
    for k in TOPKS:
        axes[0].plot(x, rej_t[f"top{k}"], marker="o", ms=4, lw=2, color=COLORS[f"top{k}"], label=f"top-{k}")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{lo:.3f}–{hi:.3f}\nn={n}" for lo, hi, n in zip(rej_t["s_it_lo"], rej_t["s_it_hi"], rej_t["n"])],
                            fontsize=8)
    axes[0].set_xlabel(f"top in-pool S_IT bucket (quantiles, K={args.vis_k})")
    axes[0].set_ylabel("mean text–text cosine")
    axes[0].set_title(f"{ts_name}: quality by top S_IT")
    axes[0].grid(alpha=0.3)
    axes[0].legend(frameon=False)
    axes[1].plot(thr["coverage"], thr["top1"], lw=2, color=COLORS["top1"], label="top-1 of accepted")
    axes[1].plot(thr["coverage"], thr["top5"], lw=2, color=COLORS["top5"], label="top-5 of accepted")
    axes[1].set_xlabel("coverage = fraction of queries accepted\n(threshold τ on top in-pool S_IT, high → low)")
    axes[1].set_title("Acceptance threshold trade-off")
    for frac in (0.25, 0.5, 0.75):
        row = thr.iloc[int(frac * nq) - 1]
        axes[1].annotate(f"τ={row['threshold']:.3f}", (row["coverage"], row["top5"]), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=8)
    axes[1].grid(alpha=0.3)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "rejection_twostage.png"), dpi=150)
    plt.close(fig)

    # manual-review export ------------------------------------------------------------
    vis_dir = os.path.join(out, "vis")
    n_vis = nq if args.n_vis is None else min(args.n_vis, nq)
    methods = {"IT": rank_it, "II": rank_ii, f"fused@{args.vis_lam:g}": vis_rank, ts_name: ts_rank}
    for u in range(n_vis):
        stem = per_query.loc[u, "image"]
        folder = os.path.join(vis_dir, stem)
        os.makedirs(folder, exist_ok=True)
        src = query_df.loc[u, "filepath"]
        shutil.copy(src, os.path.join(folder, os.path.basename(src)))
        gt_set = set(gt_caps[u])
        merged, seen, per_method = [], {}, {}
        for name, rank in methods.items():
            lst = []
            for r, j in enumerate(rank[u, :args.vis_top].tolist()):
                if name == ts_name and not ts_pool[u, j]:
                    break                                                     # pool smaller than vis_top
                cand = {
                    "rank": r + 1,
                    "caption": bank_caps[j],
                    "source_images": [os.path.basename(bank_df.loc[i, "filepath"]) for i in bank_src[j]],
                    "s_it": round(s_it[u, j].item(), 4),
                    "s_ii": round(s_ii[u, j].item(), 4),
                    "z_it": round(z_it[u, j].item(), 3),
                    "z_ii": round(z_ii[u, j].item(), 3),
                    "fused": round(fused[args.vis_lam][u, j].item(), 3),
                    "cos_mean_to_gt": round(q_mean[u, j].item(), 4),
                    "cos_max_to_gt": round(q_max[u, j].item(), 4),
                    "exact_match_gt": bank_caps[j] in gt_set,
                }
                lst.append(cand)
                if j not in seen:
                    seen[j] = {"caption": bank_caps[j], "found_by": [], "cos_mean_to_gt": cand["cos_mean_to_gt"],
                               "cos_max_to_gt": cand["cos_max_to_gt"], "exact_match_gt": cand["exact_match_gt"]}
                    merged.append(seen[j])
                seen[j]["found_by"].append(f"{name}#{r + 1}")
            per_method[name] = lst
        record = {
            "image": os.path.basename(src),
            "filepath": src,
            "ground_truth_captions": gt_caps[u],
            "r_max": round(r_max[u].item(), 3),
            "agreement": int(agreement[u]),
            "twostage_top_s_it": round(ts_top_sit[u].item(), 4),
            "twostage_pool_size": int(ts_pool[u].sum()),
            "oracle_best_bank_caption": {"caption": bank_caps[q_mean[u].argmax().item()],
                                         "cos_mean_to_gt": round(oracle_mean[u].item(), 4)},
            "candidates_merged": sorted(merged, key=lambda c: -c["cos_mean_to_gt"]),
            "candidates_by_method": per_method,
        }
        with open(os.path.join(folder, "candidates.json"), "w") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)

    summary = {
        "features": os.path.relpath(args.features, ROOT),
        "seed": args.seed,
        "query_frac": args.query_frac,
        "n_B": len(df),
        "n_query": nq,
        "n_bank_images": len(bank_df),
        "n_bank_caption_instances": len(bank_df) * len(CAPTION_COLS),
        "n_bank_captions_unique": nt,
        "n_bank_captions_multi_source": int(sum(len(s) > 1 for s in bank_src)),
        "queries_with_exact_gt_string_in_bank": int(exact.any(1).sum()),
        "vis_lam": args.vis_lam,
        "vis_k": args.vis_k,
        "n_vis": n_vis,
        "sweep": {r["method"]: {k: v for k, v in r.items() if k != "method"} for r in rows},
    }
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"queries {nq}  bank images {len(bank_df)}  unique bank captions {nt}  "
          f"(multi-source {summary['n_bank_captions_multi_source']})")
    print(f"queries whose GT string also appears in the bank: {summary['queries_with_exact_gt_string_in_bank']}/{nq}")
    cols = ["method", "pool_size", "mean_cos@1", "mean_cos@5", "mean_cos@10", "max_cos@5", "exact_match_rate@5"]
    print(sweep[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nby R_max bucket:\n" + rej_r.to_string(float_format=lambda v: f"{v:.4f}"))
    print("\nby agreement:\n" + rej_a.to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"\n{ts_name} by top in-pool S_IT:\n" + rej_t.to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
