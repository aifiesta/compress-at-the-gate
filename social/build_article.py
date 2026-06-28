"""Assemble the X (Twitter) Article: plain black and white, real animated explainers."""
import os, json, base64

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)
with open(os.path.join(ROOT, "numbers.json")) as f:
    N = json.load(f)
with open(os.path.join(ROOT, "results", "quality.json")) as f:
    Q = json.load(f)
with open(os.path.join(ROOT, "results", "latency.json")) as f:
    LAT = json.load(f)


def b64(p, mime):
    with open(p, "rb") as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode()


IMG = {
    "run": b64(os.path.join(HERE, "anim_run.gif"), "image/gif"),
    "compress": b64(os.path.join(HERE, "anim_compress.gif"), "image/gif"),
    "results": b64(os.path.join(HERE, "anim_results.gif"), "image/gif"),
    "costq": b64(os.path.join(HERE, "bw_costq.png"), "image/png"),
}

KEEP = 0.33
NEURAL = "SelectiveContext(neural)"
order = ["Lead", "Random", "TF-IDF", "TF-IDF+Q", "TextRank", "TextRank+Q",
         "Centroid+MMR", "BM25+Q", NEURAL]
lab = {NEURAL: "Selective Context (neural)"}


def trow(m):
    qr = [r for r in Q if r["method"] == m and abs(r["keep_ratio"] - KEEP) < 1e-6][0]
    fam = "neural" if m == NEURAL else "classical"
    fp = N["neuralParamMB"] if m == NEURAL else "~0"
    l = LAT[m]["1024"]["ms_median"]
    win = m in ("BM25+Q", "Centroid+MMR", "TF-IDF+Q", "TextRank+Q")
    cls = ' class="win"' if win else (' class="neu"' if fam == "neural" else "")
    return (f"<tr{cls}><td>{lab.get(m,m)}</td><td>{fam}</td>"
            f"<td>{qr['answer_retention']*100:.1f}%</td>"
            f"<td>{l:.2f} ms</td><td>{fp} MB</td></tr>")


rows = "\n".join(trow(m) for m in order)

HTML = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>We replaced a neural prompt compressor with 1970s NLP</title>
<style>
*{box-sizing:border-box}
body{margin:0;background:#ffffff;color:#000000;
  font-family:Georgia,'Times New Roman',serif;line-height:1.62;}
.wrap{max-width:680px;margin:0 auto;padding:0 22px 80px;}
.bar{display:flex;align-items:center;gap:11px;padding:16px 0 14px;border-bottom:1px solid #000;}
.av{width:36px;height:36px;background:#000;color:#fff;display:flex;align-items:center;
  justify-content:center;font-weight:700;font-family:Arial,sans-serif;}
.ha b{display:block;font-size:14px;font-family:Arial,sans-serif}
.ha span{color:#555;font-size:13px;font-family:Arial,sans-serif}
.kicker{margin-left:auto;font-size:11px;letter-spacing:.12em;font-family:monospace;color:#000;
  border:1px solid #000;padding:3px 9px;}
h1{font-size:30px;line-height:1.22;margin:26px 0 8px;font-weight:700;}
.dek{font-size:18px;color:#222;margin:0 0 6px;font-style:italic;}
.meta{font-size:13px;color:#666;font-family:Arial,sans-serif;margin:0 0 8px;}
h2{font-size:21px;margin:34px 0 8px;font-weight:700;}
p{font-size:18px;margin:13px 0;}
figure{margin:20px 0 8px;}
figure img{width:100%;border:1px solid #000;display:block;}
figcaption{font-size:13.5px;color:#444;margin-top:6px;font-family:Arial,sans-serif;}
.mono{font-family:monospace;font-size:15px;background:#f2f2f2;border:1px solid #ddd;padding:0 4px;}
.note{border:1px solid #000;padding:2px 16px;margin:20px 0;font-size:17px;}
table{width:100%;border-collapse:collapse;margin:16px 0;font-size:15px;font-family:Arial,sans-serif;}
th,td{padding:8px 10px;text-align:right;border-bottom:1px solid #ccc;}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left;}
thead th{border-bottom:2px solid #000;font-weight:700;}
tr.win td{font-weight:700;}
tr.neu td{font-style:italic;color:#444;}
hr{border:none;border-top:1px solid #000;margin:30px 0;}
.foot{font-size:13px;color:#555;font-family:Arial,sans-serif;border-top:1px solid #000;
  padding-top:14px;margin-top:34px;}
a{color:#000;}
ul{font-size:18px;} li{margin:6px 0;}
.tag{font-family:Arial,sans-serif;font-size:12px;color:#555;}
</style></head><body><div class="wrap">

<div class="bar">
  <div class="av">M</div>
  <div class="ha"><b>Mesh API Research</b><span>@meshapi</span></div>
  <div class="kicker">X ARTICLE</div>
</div>

<h1>We replaced a neural prompt compressor with NLP from the 1970s. At the gateway it was faster and more accurate.</h1>
<p class="dek">A short writeup of a result we ran this week, with the real terminal output and a walk-through of what the code actually does.</p>
<p class="meta">Mesh API routing team · June 2026 · about a 6 minute read</p>

<p>Prompt compression is having a moment. The idea is plain. Before you send a long prompt to a model, you cut the parts that do not carry much, so the model spends its limited context on the parts that do.</p>
<p>Most of the popular methods do this with a neural network. They run a small language model over your prompt to score which tokens matter. We tried that at the place where it is most tempting and most expensive, the gateway. Then we tried something much older and much cheaper. The old thing won.</p>

<h2>First, the receipts</h2>
<p>Everything below is measured on a laptop, not estimated. Here is the real run, replayed from its own log.</p>
<figure><img src="@run@" alt="terminal replay of the real benchmark run">
<figcaption>The actual stdout of the benchmark. It builds @nItems@ multi-document question-answering items from SQuAD, scores every method, times each one on this CPU, and writes the numbers to disk. About six minutes end to end.</figcaption></figure>
<p>We then ran the whole thing again on a different random seed, to check we were not getting lucky. The headline numbers moved by about a point. They held.</p>

<h2>Why the gateway changes the math</h2>
<p>A gateway is the layer that sits in front of many models and takes every request. Mesh API is one. If you put a compressor there, it runs on every call from every user. That changes what good means.</p>
<p>A neural compressor has to run a forward pass over your prompt before the real model even starts. That time lands on the critical path, so the user waits longer. The compressor also has to live somewhere, which usually means a GPU, or memory taken from the model you are actually trying to serve.</p>
<div class="note"><p>In one published benchmark, a popular neural compressor ended up with a net speedup of 0.6x. Once you count its own runtime, it made requests slower than doing nothing.</p></div>

<h2>What we did instead</h2>
<p>So we asked a boring question. How far do the old, model-free methods get? BM25. TF-IDF. TextRank. These rank sentences by how well they match the question, using word counts and simple math. No model. They run in well under a millisecond on one CPU core.</p>
<p>Here is one real item from the benchmark, compressed by BM25. Watch it score each sentence by relevance to the question, drop the low scorers, and keep the one that holds the answer.</p>
<figure><img src="@compress@" alt="BM25 compressing a real benchmark item">
<figcaption>A real item. The bars are the actual BM25 scores. The sentence the answer lives in ranks near the top, so it survives the cut. The whole step costs about a third of a millisecond and zero model weights.</figcaption></figure>

<h2>The result</h2>
<p>We ran this for every method at three times compression, then drew the real output.</p>
<figure><img src="@results@" alt="real retention and latency for every method">
<figcaption>Real answer retention (top) and latency (bottom) for every method. Black bars are query-aware classical methods. The hatched bar is the neural baseline.</figcaption></figure>
<p>The split is not classical against neural. It is query-aware against query-blind. Every method that reads the question keeps almost all the answers. TF-IDF with the question keeps @retTFIDFQ@ percent. BM25 keeps @retBMQ@. The neural method keeps @retNeural@, because it scores tokens by how surprising they are to a language model and never reads the question. A gateway request always has a question, so the expensive forward pass buys nothing here.</p>

<table><thead><tr><th>Method</th><th>Family</th><th>Answer retention</th><th>Latency</th><th>Weights</th></tr></thead>
<tbody>@ROWS@</tbody></table>
<p class="tag">Bold rows are query-aware classical methods. The italic row is the neural baseline.</p>

<figure><img src="@costq@" alt="cost versus quality">
<figcaption>Cost against quality at the gateway. Up and to the left is better. The query-aware classical methods sit there. The neural baseline is both slower and less accurate in this setting.</figcaption></figure>
<p>On cost it is not close. The classical methods sit at a fraction of a millisecond and hold no weights. The neural baseline sits at @latNeural@ milliseconds and @neuralParamMB@ megabytes.</p>

<h2>The parts we are not hiding</h2>
<p>Now the things that would get this desk-rejected if we left them out.</p>
<p>The neural baseline is small, distilgpt2 at 82 million parameters, and it ran on CPU, the same as the classical methods. That is a fair same-machine test, but on a GPU the neural latency would drop a lot. So read the @latSpeedup@ times faster as on this setup, not as a law. The accuracy gap is a separate thing. It comes from the neural method ignoring the question, and a bigger model would not fix that.</p>
<p>Our quality number is answer retention, meaning the gold answer survived the cut. It is a clean, repeatable proxy. It is not the same as end-to-end accuracy through a real model. We wrote the code to measure that through the gateway, but we have not run it yet.</p>
<p>It is our own reimplementation of the neural method, not the authors' code. The benchmark is English, with short answers. Multi-hop questions or other languages could move the line.</p>

<div class="note"><p>The point is narrow, and we think useful. Compression has been studied as a modeling problem. At the gateway it is a systems problem first. The thing on the critical path of every request should be cheap before it is clever. Here the cheap thing was also the accurate one.</p></div>

<hr>
<p>The full preprint, the benchmark, and all the code are open. Everything reproduces offline. The only network call is a one-time download of SQuAD and distilgpt2.</p>
<ul>
<li>Paper (PDF): <a href="#">link</a></li>
<li>Benchmark and code: <a href="#">link</a></li>
</ul>

<div class="foot">Measured on an Apple M3 Pro, CPU. @nItems@ multi-document QA items, mean @meanSrcTokens@ tokens each. BM25 with the question: @retBMQ@ percent retention, @latBMQ@ ms, 0 MB. Neural Selective Context: @retNeural@ percent, @latNeural@ ms, @neuralParamMB@ MB.<br>
Mesh API. One API, every model. meshapi.ai</div>

</div></body></html>"""

out = HTML.replace("@ROWS@", rows)
for k, v in IMG.items():
    out = out.replace("@" + k + "@", v)
for k, v in N.items():
    out = out.replace("@" + k + "@", str(v))

with open(os.path.join(HERE, "x_article.html"), "w") as f:
    f.write(out)
print("wrote x_article.html (black & white, with real animations)")
