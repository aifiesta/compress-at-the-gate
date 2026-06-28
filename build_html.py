"""
Render the paper to a print-ready HTML (academic single-column, serif), reading
the real numbers from numbers.json and embedding the PNG figures as base64 so the
file is fully self-contained. A PDF is produced from this HTML via headless Chrome.
"""
import os, json, base64

HERE = os.path.dirname(__file__)
with open(os.path.join(HERE, "numbers.json")) as f:
    N = json.load(f)
with open(os.path.join(HERE, "results", "quality.json")) as f:
    Q = json.load(f)
with open(os.path.join(HERE, "results", "latency.json")) as f:
    LAT = json.load(f)
with open(os.path.join(HERE, "results", "meta.json")) as f:
    META = json.load(f)


def b64(path):
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


KEEP = 0.33
NEURAL = "SelectiveContext(neural)"
methods = ["Lead", "Random", "TF-IDF", "TF-IDF+Q", "TextRank", "TextRank+Q",
           "Centroid+MMR", "BM25+Q", NEURAL]
label = {NEURAL: "Selective Context (neural)"}


def row(m):
    qr = [r for r in Q if r["method"] == m and abs(r["keep_ratio"] - KEEP) < 1e-6][0]
    fam = "neural" if m == NEURAL else "classical"
    fp = N["neuralParamMB"] if m == NEURAL else "≈0"
    l = LAT[m]["1024"]["ms_median"]
    bold = m in ("BM25+Q", "Centroid+MMR")
    name = label.get(m, m)
    cls = ' class="hl"' if bold else ""
    return (f"<tr{cls}><td>{name}</td><td>{fam}</td>"
            f"<td>{qr['answer_retention']*100:.1f}</td>"
            f"<td>{qr['mean_token_reduction']*100:.1f}</td>"
            f"<td>{l:.2f}</td><td>{fp}</td></tr>")


rows = "\n".join(row(m) for m in methods)
figs = {k: b64(os.path.join(HERE, "figures", k + ".png"))
        for k in ["fig_latency", "fig_retention", "fig_cost_quality", "fig_footprint"]}

HTML = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>Compress-at-the-Gate</title>
<style>
@page {{ size: A4; margin: 18mm 16mm; }}
body {{ font-family: 'Georgia','Times New Roman',serif; color:#111; line-height:1.5;
  font-size:10.5pt; max-width: 820px; margin:0 auto; }}
h1 {{ font-size:19pt; text-align:center; line-height:1.25; margin:0 0 4px; }}
.authors {{ text-align:center; color:#333; margin-bottom:2px; }}
.affil {{ text-align:center; color:#555; font-size:9.5pt; margin-bottom:14px; }}
h2 {{ font-size:12.5pt; border-bottom:1.5px solid #6F4BD8; padding-bottom:2px; margin-top:18px; color:#2a2150; }}
h3 {{ font-size:10.8pt; margin:12px 0 2px; color:#3a2f66; }}
.abstract {{ background:#f7f6fb; border-left:3px solid #6F4BD8; padding:12px 16px; font-size:10pt; margin:10px 0 6px; }}
.abstract b {{ color:#4b2fd0; }}
code,.mono {{ font-family:'SF Mono',Menlo,monospace; font-size:9pt; }}
table {{ border-collapse:collapse; width:100%; font-size:9.3pt; margin:8px 0; }}
th,td {{ border-bottom:1px solid #ccc; padding:4px 7px; text-align:right; }}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2) {{ text-align:left; }}
thead th {{ border-bottom:1.5px solid #333; }}
tr.hl td {{ background:#efeafc; font-weight:bold; }}
.figrow {{ display:flex; gap:12px; margin:12px 0; align-items:flex-start; }}
.figrow figure {{ margin:0; flex:1; }}
figure img {{ width:100%; border:1px solid #eee; border-radius:4px; }}
figcaption {{ font-size:8.6pt; color:#444; margin-top:3px; }}
.eq {{ text-align:center; font-style:italic; margin:8px 0; color:#222; }}
ol,ul {{ margin:4px 0 8px 18px; }} li {{ margin:2px 0; }}
.refs {{ font-size:8.7pt; line-height:1.4; }} .refs li {{ margin:2px 0; }}
.tag {{ display:inline-block; background:#6F4BD8; color:#fff; font-size:8pt; padding:1px 7px;
  border-radius:8px; font-family:sans-serif; letter-spacing:.04em; }}
small.note {{ color:#666; }}
</style></head><body>

<p style="text-align:center"><span class="tag">PREPRINT · JUNE 2026</span></p>
<h1>Compress-at-the-Gate: Query-Aware Classical Prompt Compression<br>
as a Zero-Footprint Context-Engineering Primitive for LLM Gateways</h1>
<div class="authors">Raushan</div>
<div class="affil">Mesh API / AI Fiesta · raushan@aifiesta.ai<br>
<small class="note">Author list is a placeholder for the camera-ready version.</small></div>

<div class="abstract">
<b>Abstract.</b> LLM <i>gateways</i>—unified APIs that route one request format to hundreds
of backing models—sit on the critical path of a large fraction of production LLM traffic.
Because a gateway touches <i>every</i> request, any context-engineering transformation it
applies must be cheap in two currencies the prompt-compression literature largely ignores:
<i>tail latency</i> and <i>accelerator footprint</i>. State-of-the-art neural compressors
(Selective Context, LLMLingua, RECOMP) estimate token importance with a language-model
forward pass, which needs its own GPU and can make end-to-end inference <i>slower</i>. We
ask how far <i>classical</i>, model-free extractive compression goes at the gate. Query-aware
lexical methods (BM25, centroid+MMR) retain <b>{retBMQ}%</b> of gold answers at ~3× compression
—<b>{retGainNeuralX}× the {retNeural}%</b> of a faithful neural Selective-Context baseline—while
running in <b>{latBMQ} ms</b> vs <b>{latNeural} ms</b> per request (<b>{latSpeedup}× faster</b>)
and holding <i>zero</i> resident model weights against the baseline's {neuralParamMB} MB. At the
gateway, the classical/neural trade-off inverts.
</div>

<h2>1. Introduction</h2>
<p>Building LLM applications has shifted from <i>prompt engineering</i> to <i>context
engineering</i>: deciding what an agent "knows, sees, and remembers at the moment of
action" [1]. A central operation is <i>compression</i>—removing low-value tokens so more of a
finite, expensive context window carries signal, mitigating cost and the "lost-in-the-middle"
degradation [2].</p>
<p>Almost all recent work frames compression as a <i>modeling</i> problem. Selective Context
[3] prunes lexical units with low self-information under a small LM; LLMLingua / LongLLMLingua
[4,5] use (contrastive) perplexity from a 0.5–7B LM; LLMLingua-2 [6] distills a token
classifier; RECOMP [7] trains dedicated compressors. All share a structural cost: <b>a neural
forward pass over the prompt at compression time.</b></p>
<p>That cost is acceptable inside one application; it is not obviously acceptable at the <i>LLM
gateway</i>. A gateway such as Mesh API exposes one OpenAI-compatible endpoint fronting 300+
models [15], applies routing, fallback and templating, and processes <i>every</i> request for
<i>every</i> tenant. There a compressor inherits three constraints the literature rarely
measures:</p>
<ol>
<li><b>Critical-path latency.</b> Compression latency adds directly to time-to-first-token. In
the LongLLMLingua evaluation, Selective Context's <i>net</i> speedup is 0.6×—it makes the request
<i>slower</i> [5].</li>
<li><b>Accelerator footprint.</b> A multi-tenant, horizontally-scaled gateway paying for a
dedicated compressor GPU (or stolen VRAM) multiplies fleet cost.</li>
<li><b>Tail behavior.</b> Neural-compressor latency grows with prompt length and contends with
generation for the accelerator, widening p95/p99.</li>
</ol>
<p>We revisit <i>classical</i>, model-free extractive compression—TextRank [8], LexRank [9],
centroid [10], MMR [11], BM25 [12]—as a <i>systems</i> primitive for the gate. Our central,
somewhat counterintuitive finding: at realistic gateway ratios, <i>query-aware</i> classical
methods do not merely approach the neural baseline—they <b>exceed</b> a faithful
Selective-Context reimplementation on answer retention, because that method is
query-<i>agnostic</i> while a gateway request almost always carries an explicit query.</p>

<h2>2. Related Work</h2>
<h3>Neural prompt compression</h3>
<p>Selective Context [3] removes low-self-information units. LLMLingua [4] adds a budget
controller and iterative perplexity pruning; LongLLMLingua [5] adds question-aware
coarse-to-fine compression, reordering and subsequence recovery; LLMLingua-2 [6] swaps
perplexity for a distilled classifier (3–6× faster). RECOMP [7] trains compressors for RAG.
All require a forward pass at compression time. Notably, in [5] simple lexical baselines (BM25,
Gzip) outranked Selective Context on coarse selection—a hint we make systematic.</p>
<h3>Classical extractive summarization & ranking</h3>
<p>TextRank [8] and LexRank [9] score sentences by graph centrality; centroid methods [10] by
proximity to a document centroid; MMR [11] trades relevance against redundancy; BM25 [12] is
the canonical lexical query-relevance function.</p>
<h3>LLM routing & gateways</h3>
<p>Routers like RouteLLM [13] pick a model per query; gateways [15] add a unified API, fallback
and templating. We study a complementary transformation—<i>what context</i> reaches the model—
co-located at the same layer. Co-location is practical because serving stacks
(PagedAttention/vLLM [14], FlashAttention) already saturate the GPU with generation; a CPU-side
compressor adds no accelerator contention.</p>

<h2>3. The Gateway Compression Problem</h2>
<p>A request is (q, C) with query q and context C of sentences. A compressor f returns
C′=f(q,C) with token budget |C′| ≤ β|C|. The application objective maximizes downstream
quality U for a fixed prompt. The gateway objective differs: hold an SLA over a heterogeneous,
multi-tenant stream while minimizing fleet cost. With ℓ<sub>f</sub> the added latency and
m<sub>f</sub> the resident accelerator memory:</p>
<p class="eq">min<sub>f</sub> E[cost(C′)] s.t. ℓ<sub>f</sub>(|C|) ≤ τ (critical path),
m<sub>f</sub> ≤ μ (per-replica VRAM), U(q,C′) ≥ U<sub>min</sub>.</p>
<p>Neural compressors optimize U aggressively but treat ℓ<sub>f</sub> and m<sub>f</sub> as free.
At the gate they are not. Classical f have m<sub>f</sub> ≈ 0 and ℓ<sub>f</sub> of microseconds–
milliseconds; the empirical question is whether they meet U<sub>min</sub>.</p>

<h2>4. Method</h2>
<p>We evaluate model-free extractive compressors that select a subset of sentences and restore
source order, query-aware when q is present (the gateway case): <b>Lead/Random</b> baselines;
<b>TF-IDF(+Q)</b> salience; <b>TextRank(+Q)</b> PageRank over a TF-IDF cosine graph with a query
personalization vector; <b>Centroid+MMR</b> greedy query/centroid relevance vs redundancy;
<b>BM25+Q</b> Okapi sentence relevance. <b>Neural baseline:</b> a faithful Selective Context [3]—
per-token NLL under <span class="mono">distilgpt2</span> in a sliding 512-token window,
aggregated to per-sentence self-information; query-agnostic by design, requiring a forward pass
over the whole prompt. At the gateway, f runs as a stateless CPU stage between request parsing
and model dispatch, adding no GPU contention.</p>

<h2>5. Experimental Setup</h2>
<p><b>Benchmark.</b> A multi-document QA set from SQuAD v1.1 [16]: each item concatenates the gold
paragraph with {nDistractors} distractor paragraphs from unrelated articles, gold placed in the
middle to stress "lost-in-the-middle" [2]. {nItems} items, mean {meanSrcTokens} tokens over
{meanPassages} passages. <b>Metrics:</b> answer retention (compressed context still contains the
SQuAD-normalized gold span; intrinsic, LLM-free), token reduction (cl100k_base), latency (median
ms/request over {{256…4096}} tokens), footprint (resident weights). All timing on one
{machine} CPU; every method, including the neural baseline, is timed on the same hardware.</p>

<h2>6. Results</h2>
<p><b>Quality — query-awareness is the discriminator.</b> Every query-aware classical method
retains far more answers than every query-agnostic one. TF-IDF+Q ({retTFIDFQ}%), TextRank+Q
({retTRQ}%), BM25+Q ({retBMQ}%) and Centroid+MMR ({retMMR}%) all retain ≳96% of gold answers at
~3× compression, versus {retNeural}% for the neural Selective-Context baseline — which, being
query-agnostic, barely beats query-agnostic TF-IDF ({retTFIDF}%) and the Lead baseline
({retLead}%). The lesson is not "neural vs classical" but "query-aware vs query-agnostic": a
gateway request carries an explicit query, and cheaply conditioning sentence selection on it
recovers almost all answer-bearing content. The expensive neural forward pass buys nothing here
because Selective Context does not see the query.</p>

{TABLE}
<small class="note">Adding the query signal lifts TF-IDF from {retTFIDF}% to {retTFIDFQ}% and
TextRank from its query-agnostic score to {retTRQ}% retention — the dominant factor is
query-conditioning, not the choice of classical scorer.</small>

<div class="figrow">
<figure><img src="{fig_latency}"><figcaption><b>Fig 1.</b> Latency scales with context
length; classical methods stay in the single-digit-ms band, the neural compressor in the
hundreds of ms ({machine}, CPU).</figcaption></figure>
<figure><img src="{fig_retention}"><figcaption><b>Fig 2.</b> Answer retention vs token
reduction. Query-aware lexical methods retain answers far deeper into compression.</figcaption></figure>
</div>
<div class="figrow">
<figure><img src="{fig_cost_quality}"><figcaption><b>Fig 3.</b> Cost–quality frontier @1024
tokens / ~3× compression. Top-left is better: cheap and faithful.</figcaption></figure>
<figure><img src="{fig_footprint}"><figcaption><b>Fig 4.</b> Resident model weights: classical
≈0 MB vs {neuralParamMB} MB.</figcaption></figure>
</div>

<p><b>Latency & footprint.</b> Classical methods compress a 1024-token prompt in
{latRangeLo}–{latRangeHi} ms; the neural baseline needs {latNeural} ms ({latSpeedup}× slower) and
grows with prompt length. The baseline holds {neuralParamMB} MB of weights ({neuralParams}
params); classical methods hold none.</p>
<p><b>Fleet-cost projection.</b> At R req/s with mean 1024-token prompts, a neural compressor at
{latNeural} ms needs ≈ one busy accelerator core per ~{neuralRPS} req/s plus {neuralParamMB} MB
VRAM per replica; the classical stage absorbs the same load on a fraction of a CPU core with no
VRAM—the difference between a compression tier that is a rounding error and one that rivals
serving itself.</p>

<h2>7. Discussion</h2>
<p>This is not a claim that classical beats neural compression in general—LongLLMLingua's
question-aware perplexity remains stronger for aggressive, abstractive-leaning compression and
diffuse answers [5,6]. The claim is narrower: <b>at the gateway—where latency is on the critical
path, footprint multiplies across a fleet, and an explicit query is almost always present—the
trade-off inverts</b>, and a model-free query-aware extractor is the right default. A natural
hybrid runs the classical stage on every request and escalates to a neural compressor only for
the few long, query-diffuse prompts a cheap signal flags—mirroring model-router escalation [13].</p>
<p><b>Limitations.</b> (i) The metric is intrinsic answer-span retention—a strong proxy for
extractive faithfulness, not a substitute for end-to-end accuracy (the harness supports an
optional gateway-backed eval). (ii) Methods are extractive, sentence-level. (iii) The benchmark
is English SQuAD with localized answers; multi-hop/non-English may shift the frontier. (iv) We
compare against query-agnostic Selective Context; a query-aware neural compressor would narrow
the quality gap at, by construction, even higher latency and footprint.</p>

<h2>8. Conclusion</h2>
<p>Prompt compression has been studied almost entirely as a modeling problem. Viewed as a
<i>systems</i> problem at the gateway—the layer that touches every request—the priorities
reorder: a compressor must first be cheap in latency and footprint, then maximize retention. In
that regime, decades-old query-aware extractive methods retain {retBMQ}% of answers at ~3×
compression while running {latSpeedup}× faster than a neural baseline with zero resident weights.
The cheapest context-engineering primitive may also be the best one to put at the gate.</p>

<h2>References</h2>
<ol class="refs">
<li>The Context Engineering Survey: From Prompts to Multi-Agent Architectures. arXiv:2603.09619, 2026.</li>
<li>N. F. Liu et al. Lost in the Middle. TACL 2024. arXiv:2307.03172.</li>
<li>Y. Li et al. Compressing Context to Enhance Inference Efficiency (Selective Context). EMNLP 2023. arXiv:2304.12102.</li>
<li>H. Jiang et al. LLMLingua. EMNLP 2023. arXiv:2310.05736.</li>
<li>H. Jiang et al. LongLLMLingua. ACL 2024. arXiv:2310.06839.</li>
<li>Z. Pan et al. LLMLingua-2. Findings of ACL 2024. arXiv:2403.12968.</li>
<li>F. Xu, W. Shi, E. Choi. RECOMP. ICLR 2024. arXiv:2310.04408.</li>
<li>R. Mihalcea, P. Tarau. TextRank. EMNLP 2004.</li>
<li>G. Erkan, D. Radev. LexRank. JAIR 2004.</li>
<li>D. Radev et al. Centroid-based Summarization. IP&M 2004.</li>
<li>J. Carbonell, J. Goldstein. Maximal Marginal Relevance. SIGIR 1998.</li>
<li>S. Robertson, H. Zaragoza. BM25 and Beyond. FnTIR 2009.</li>
<li>I. Ong et al. RouteLLM. arXiv:2406.18665, 2024.</li>
<li>W. Kwon et al. PagedAttention / vLLM. SOSP 2023.</li>
<li>Mesh API. Product Overview & Auto-Routing Docs. developers.meshapi.ai, 2026.</li>
<li>P. Rajpurkar et al. SQuAD. EMNLP 2016. arXiv:1606.05250.</li>
</ol>
</body></html>"""

out = HTML.format(TABLE=("<table><thead><tr><th>Method</th><th>Family</th>"
                         "<th>Answer ret. (%)</th><th>Token red. (%)</th>"
                         "<th>Latency (ms)</th><th>Footprint (MB)</th></tr></thead>"
                         f"<tbody>{rows}</tbody></table>"),
                  **figs, **N)
with open(os.path.join(HERE, "paper.html"), "w") as f:
    f.write(out)
print("wrote paper.html")
