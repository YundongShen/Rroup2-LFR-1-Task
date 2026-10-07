"""Build the course-mandated A/B/C manifests from the raw RSICD files.

Outputs (cfg.out_dir):
  A_unlabeled.csv   split A (orig train): filepath only -- no captions/labels (course rule)
  B_labeled.csv     split B (orig val):   filepath, label, caption_0..4
  C_test.csv        split C (orig test):  filepath, label, caption_0..4 -- final evaluation only
  classes.json      class index -> folder name

The train/val division *inside* B is not done here; it is generated per run
from the `data` config (see src/data.py).

Usage: python scripts/prepare_data.py [raw_dir=...] [out_dir=...]
"""
import json
import os

import hydra
import pandas as pd
from omegaconf import DictConfig


@hydra.main(config_path="../configs", config_name="prepare", version_base="1.3")
def main(cfg: DictConfig):
    file2cls = {}
    for t in sorted(os.listdir(cfg.classes_dir)):
        for line in open(os.path.join(cfg.classes_dir, t), encoding="utf-8", errors="replace"):
            if line.strip():
                file2cls[line.strip()] = t[:-4]
    classes = sorted(set(file2cls.values()))
    cidx = {c: i for i, c in enumerate(classes)}

    meta = json.load(open(cfg.captions_json))["images"]
    rows = {"train": [], "val": [], "test": []}
    for im in meta:
        path = os.path.join(cfg.images_dir, im["filename"])
        if im["split"] == "train":  # split A: image path only (no captions, no class label)
            rows["train"].append({"filepath": path})
            continue
        r = {"filepath": path, "label": cidx[file2cls[im["filename"]]]}
        for k, s in enumerate(im["sentences"]):
            r[f"caption_{k}"] = s["raw"].strip()
        rows[im["split"]].append(r)

    os.makedirs(cfg.out_dir, exist_ok=True)
    pd.DataFrame(rows["train"]).to_csv(f"{cfg.out_dir}/A_unlabeled.csv", index=False)
    pd.DataFrame(rows["val"]).to_csv(f"{cfg.out_dir}/B_labeled.csv", index=False)
    pd.DataFrame(rows["test"]).to_csv(f"{cfg.out_dir}/C_test.csv", index=False)
    json.dump([{"idx": i, "name": c} for i, c in enumerate(classes)],
              open(f"{cfg.out_dir}/classes.json", "w"), indent=1)
    print({"A": len(rows["train"]), "B": len(rows["val"]), "C": len(rows["test"]), "classes": len(classes)})


if __name__ == "__main__":
    main()
