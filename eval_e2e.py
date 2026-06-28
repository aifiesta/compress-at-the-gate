"""
OPTIONAL end-to-end evaluation through the Mesh API gateway.

Measures downstream QA accuracy (SQuAD Exact-Match / F1) when the *compressed*
context is sent to a real model behind the OpenAI-compatible Mesh API endpoint,
versus the uncompressed context. This turns the intrinsic answer-retention metric
into a true task-accuracy number.

It is intentionally decoupled from the main benchmark: the headline results in the
paper are reproducible WITHOUT any API key. Run this only to obtain end-to-end
numbers.

Usage:
    export MESH_API_KEY=mesh_sk_...
    python eval_e2e.py --model gpt-4o-mini --method BM25+Q --keep 0.33 --n 100
"""
from __future__ import annotations
import os, sys, argparse, time, json, re, string

from compressors import CLASSICAL
from benchmark import build_multidoc_squad, normalize_answer


def f1(pred: str, gold: str) -> float:
    p = normalize_answer(pred).split()
    g = normalize_answer(gold).split()
    if not p or not g:
        return float(p == g)
    common = {}
    for t in p:
        common[t] = min(p.count(t), g.count(t))
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    prec = num_same / len(p)
    rec = num_same / len(g)
    return 2 * prec * rec / (prec + rec)


def em(pred: str, gold: str) -> float:
    return float(normalize_answer(gold) in normalize_answer(pred))


def ask(client, model, context, question):
    msg = [
        {"role": "system", "content": "Answer the question using only the context. "
                                      "Reply with the shortest exact answer span."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"},
    ]
    r = client.chat.completions.create(model=model, messages=msg,
                                       temperature=0, max_tokens=32)
    return r.choices[0].message.content.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--method", default="BM25+Q", choices=list(CLASSICAL))
    ap.add_argument("--keep", type=float, default=0.33)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--base-url", default="https://api.meshapi.ai/v1")
    args = ap.parse_args()

    key = os.environ.get("MESH_API_KEY")
    if not key:
        print("MESH_API_KEY not set. This optional evaluation needs a gateway key.")
        print("The paper's headline results do NOT require it; see run_all.py.")
        sys.exit(2)

    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key=key)
    items = build_multidoc_squad(n_items=args.n, n_distractors=7, seed=13)
    method = CLASSICAL[args.method]()

    agg = {"orig_em": 0, "orig_f1": 0.0, "comp_em": 0, "comp_f1": 0.0,
           "orig_tokens": 0, "comp_tokens": 0, "n": 0}
    from benchmark import n_tokens
    for it in items:
        comp = method.compress(it.context, it.question, keep_ratio=args.keep).text
        try:
            a_orig = ask(client, args.model, it.context, it.question)
            a_comp = ask(client, args.model, comp, it.question)
        except Exception as e:
            print("API error:", str(e)[:120]); continue
        agg["orig_em"] += em(a_orig, it.answer); agg["orig_f1"] += f1(a_orig, it.answer)
        agg["comp_em"] += em(a_comp, it.answer); agg["comp_f1"] += f1(a_comp, it.answer)
        agg["orig_tokens"] += it.src_tokens; agg["comp_tokens"] += n_tokens(comp)
        agg["n"] += 1

    n = max(1, agg["n"])
    out = {
        "model": args.model, "method": args.method, "keep": args.keep, "n": n,
        "orig_EM": agg["orig_em"]/n*100, "orig_F1": agg["orig_f1"]/n*100,
        "comp_EM": agg["comp_em"]/n*100, "comp_F1": agg["comp_f1"]/n*100,
        "token_reduction_pct": (1 - agg["comp_tokens"]/max(1, agg["orig_tokens"]))*100,
    }
    os.makedirs("results", exist_ok=True)
    with open("results/e2e.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
