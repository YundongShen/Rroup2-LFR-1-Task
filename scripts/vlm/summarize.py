"""Print the VLM-caption experiment tables from runs/<exp_name>/<ts>/metrics.json (latest run per name).

    python scripts/vlm/summarize.py
"""
import glob
import json

ROWS = [
    ("pilot_zeroshot", "Zero-shot (no training)"),
    ("pilot_b_vlm", "B-train, Gemma caption (1/img)"),
    ("pilot_b_real1", "B-train, 1 human caption/img"),
    ("pilot_b_real", "B-train, 5 human captions/img"),
    ("b_real_a_vlm_w0.5", "B human + A Gemma, lambda=0.5"),
    ("b_real_a_vlm_w1.0", "B human + A Gemma, lambda=1.0"),
]
KEYS = ["zs_acc", "zs_mean_per_class_acc", "i2t_R@1", "t2i_R@1", "i2t_R@5", "t2i_R@5", "mR"]

print("| Method | " + " | ".join(KEYS) + " |")
print("|---" * (len(KEYS) + 1) + "|")
for name, label in ROWS:
    files = sorted(glob.glob(f"runs/{name}/*/metrics.json"))
    if not files:
        continue
    s = json.load(open(files[-1]))["summary"]
    print(f"| {label} | " + " | ".join(f"{100 * s[k]['mean']:.1f} ± {100 * s[k]['std']:.1f}" for k in KEYS) + " |")
