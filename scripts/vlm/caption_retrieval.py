"""How discriminative are VLM captions? Zero-shot CLIP retrieval on split B, real vs synthetic captions.

Each image gets exactly one query caption per condition, so galleries have equal size (1094):
  real_0   : first human caption
  real_rnd : a random human caption (seeded)
  vlm      : the generated caption
Reports t2i and i2t R@1/5/10. Uses only split B (allowed for training/analysis); never touches C.
"""
import argparse, json, random
import open_clip, torch, torch.nn.functional as F
from PIL import Image


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--captions", required=True)
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="laion2b_s34b_b79k")
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()

    recs = {r["filename"]: r for r in json.load(open(f"{a.data}/dataset_rsicd.json"))["images"] if r["split"] == "val"}
    vlm = {json.loads(l)["filename"]: json.loads(l)["caps"][0] for l in open(a.captions)}
    files = sorted(set(recs) & set(vlm))
    rng = random.Random(0)
    conds = {
        "real_0": [recs[f]["sentences"][0]["raw"] for f in files],
        "real_rnd": [rng.choice(recs[f]["sentences"])["raw"] for f in files],
        "vlm": [vlm[f] for f in files],
    }

    model, _, pre = open_clip.create_model_and_transforms(a.model, pretrained=a.pretrained, device=a.device)
    tok = open_clip.get_tokenizer(a.model)
    model.eval()
    img = []
    for i in range(0, len(files), 128):
        x = torch.stack([pre(Image.open(f"{a.data}/RSICD_images/{f}").convert("RGB")) for f in files[i:i + 128]])
        img.append(F.normalize(model.encode_image(x.to(a.device)).float(), dim=-1))
    img = torch.cat(img)

    gt = torch.arange(len(files), device=a.device)
    print(f"{a.model}/{a.pretrained}  n={len(files)}")
    for name, caps in conds.items():
        txt = torch.cat([F.normalize(model.encode_text(tok(caps[i:i + 256]).to(a.device)).float(), dim=-1)
                         for i in range(0, len(caps), 256)])
        sim = img @ txt.T
        row = []
        for d, s in (("i2t", sim), ("t2i", sim.T)):
            rank = (s > s[gt, gt][:, None]).sum(1)
            row += [f"{d} R@{k}={100 * (rank < k).float().mean():.1f}" for k in (1, 5, 10)]
        print(f"  {name:9s} " + "  ".join(row))


if __name__ == "__main__":
    main()
