"""Caption RSICD images with a local Ollama VLM. Resumable JSONL output.

Rules respected:
  * split A ('train') captions are never read; only B ('val') captions are used as style examples.
  * filenames (which contain the class name) are never sent to the model.
"""
import argparse, base64, json, random, re, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen

PROMPTS = {
    "v1": """You are captioning an aerial / satellite remote sensing image.
Write {n} different short captions (8-20 words each) in the style of these examples:
{examples}
Describe only what is clearly visible: land use, main objects, layout, colours. Do not guess counts unless obvious.
Output exactly {n} lines, one caption per line, lowercase, no numbering.""",
    # v2: force the scene-level noun (what the human captions lead with), without giving any class list.
    "v2": """You are captioning an aerial / satellite remote sensing image.
First decide what kind of place this is as a whole and what its most prominent structure is
(for example a bridge over a river, a port, an airport, a farmland, a residential area, a stadium).
Then write {n} different short captions (8-20 words each) that each mention that scene or structure explicitly,
in the style of these examples:
{examples}
Describe only what is clearly visible. Do not guess counts unless obvious.
Output only the {n} captions, one per line, lowercase, no numbering, no reasoning.""",
}


def style_examples(records, k, seed):
    caps = [s["raw"] for r in records if r["split"] == "val" for s in r["sentences"]]
    caps = sorted(set(caps))
    return random.Random(seed).sample(caps, k)


def caption(img_path, prompt, model, host, n):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt,
                      "images": [base64.b64encode(img_path.read_bytes()).decode()]}],
        "stream": False, "think": False,
        "options": {"temperature": 0.7, "num_predict": 40 * n},
    }
    t0 = time.time()
    req = Request(f"{host}/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
    out = json.load(urlopen(req, timeout=600))
    lines = [re.sub(r"^[\s\-\*\d\.\)]+", "", l).strip() for l in out["message"]["content"].splitlines()]
    return {"caps": [l for l in lines if l][:n], "sec": time.time() - t0,
            "prompt_tok": out.get("prompt_eval_count"), "gen_tok": out.get("eval_count"),
            "gen_tps": out.get("eval_count", 0) / max(out.get("eval_duration", 1) / 1e9, 1e-9)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dir with dataset_rsicd.json and RSICD_images/")
    ap.add_argument("--split", default="val", choices=["train", "val"])  # never 'test'
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--n", type=int, default=1, help="captions per image")
    ap.add_argument("--examples", type=int, default=3)
    ap.add_argument("--prompt", default="v2", choices=list(PROMPTS))
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--model", default="gemma4:12b-mlx")
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    data = Path(a.data)
    records = json.load(open(data / "dataset_rsicd.json"))["images"]
    ex = style_examples(records, a.examples, a.seed)
    prompt = PROMPTS[a.prompt].format(n=a.n, examples="\n".join(f"- {c}" for c in ex))

    todo = [r for r in records if r["split"] == a.split]
    random.Random(a.seed).shuffle(todo)
    todo = todo[: a.limit]
    out = Path(a.out)
    done = {json.loads(l)["filename"] for l in out.open()} if out.exists() else set()
    todo = [r for r in todo if r["filename"] not in done]
    print(f"{len(todo)} images to caption ({len(done)} already done)", file=sys.stderr)

    t0 = time.time()
    with out.open("a") as f, ThreadPoolExecutor(a.parallel) as pool:
        futs = [(r, pool.submit(caption, data / "RSICD_images" / r["filename"], prompt, a.model, a.host, a.n))
                for r in todo]
        for i, (r, fu) in enumerate(futs, 1):
            res = fu.result()
            f.write(json.dumps({"filename": r["filename"], **res}) + "\n"); f.flush()
            el = time.time() - t0
            print(f"[{i}/{len(todo)}] {res['sec']:.1f}s gen {res['gen_tok']} tok @ {res['gen_tps']:.1f} tok/s"
                  f" | avg {el / i:.2f}s/img | {res['caps'][:1]}", file=sys.stderr)
    print(f"wall {time.time() - t0:.1f}s for {len(todo)} imgs", file=sys.stderr)


if __name__ == "__main__":
    main()
