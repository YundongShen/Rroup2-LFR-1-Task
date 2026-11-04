"""Content-word overlap of generated captions vs the human B captions (no class labels used).
For each generated caption: precision/recall/F1 of content words vs the union of the image's refs."""
import json, re, sys, statistics as st

STOP = set("""a an the is are of in on at to and with by near next some many several there it its this that
which be been being as from into around two three four five one other others side sides very lot lots
of has have area areas""".split())


def words(s):
    w = re.findall(r"[a-z]+", s.lower())
    w = [x[:-1] if x.endswith("s") and len(x) > 3 else x for x in w]  # crude plural folding
    return {x for x in w if x not in STOP}


def main(gt_json, *runs):
    gt = {r["filename"]: [s["raw"] for s in r["sentences"]] for r in json.load(open(gt_json))["images"]}
    for run in runs:
        P, R, F = [], [], []
        for l in open(run):
            x = json.loads(l)
            ref = set().union(*(words(c) for c in gt[x["filename"]]))
            for c in x["caps"][:1]:
                g = words(c)
                p = len(g & ref) / max(len(g), 1); r = len(g & ref) / max(len(ref), 1)
                P.append(p); R.append(r); F.append(2 * p * r / max(p + r, 1e-9))
        print(f"{run}: n={len(F)}  P={st.mean(P):.3f}  R={st.mean(R):.3f}  F1={st.mean(F):.3f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
