# X / Twitter Thread: "We replaced a neural prompt compressor with 1970s NLP"

**Goal:** authority + traffic to the paper · **Audience:** AI / dev / founder X · **Length:** 11 tweets
**Media plan (all real, all black/white):**
- T1 → `anim_results.gif` (the headline bars)
- T5 → `anim_run.gif` (the real terminal run)
- T6 → `anim_compress.gif` (BM25 compressing a real item)
- T8 → `bw_costq.png` (cost vs quality)

Post the GIFs as native video/GIF uploads. No em dashes anywhere, plain language.

---

### Thread

**1/ (HOOK, attach `anim_results.gif`)**
We replaced a 312 MB neural prompt compressor with NLP from the 1970s.

At the LLM gateway it was 1213x faster AND more accurate.

Same machine. Same benchmark. The numbers are real and I'll show you the terminal. 🧵

**2/**
Prompt compression is having a moment.

Before you send a long prompt to a model, you cut the parts that do not carry much, so the context window holds more signal.

Most popular methods do this with a neural network. That is the part that breaks at scale.

**3/**
LLMLingua, Selective Context, RECOMP and friends all score tokens by running a small language model over your prompt.

Good results in papers.

But every one of them needs a neural forward pass before the real model even starts.

**4/**
That is fine inside one app. It is a problem at the gateway.

A gateway sits in front of many models and takes every request from every user.

A compressor there pays for itself on every single call. Latency and GPU memory both bite.

**5/ (attach `anim_run.gif`)**
First, proof. This is the actual run, replayed from its own log.

It builds 200 multi-doc QA items from SQuAD, scores every method, times each on a laptop CPU, writes the numbers to disk.

About 6 minutes. I ran it twice on different seeds. It held.

**6/ (attach `anim_compress.gif`)**
Here is what the cheap method does, on one real item.

BM25 scores each sentence by how well it matches the question, drops the low scorers, keeps the one with the answer.

The bars are real scores. Cost: about 0.3 ms and zero model weights.

**7/**
The result was not classical vs neural.

It was query-aware vs query-blind.

Every method that reads the question kept almost all the answers. The neural one kept 38.7%, because it never looks at the question. A gateway request always has one.

**8/ (attach `bw_costq.png`)**
Numbers at 3x compression:

BM25 + question: 96.5% answers kept, 0.33 ms, 0 MB
TF-IDF + question: 98.5%
Neural Selective Context: 38.7%, 406 ms, 312 MB

Cheaper and more accurate. Up and to the left is better.

**9/ (the honest part)**
What I am not hiding:

The neural baseline is small (distilgpt2) and ran on CPU, same as the rest. On a GPU its latency drops a lot, so read 1213x as "on this setup."

The accuracy gap is separate. It comes from ignoring the question. A bigger model does not fix that.

**10/**
More caveats, because they matter:

Our metric is whether the answer survived the cut, not full end-to-end accuracy yet.

It is my reimplementation of the neural method. English, short answers.

The claim is narrow. At the gateway, cheap query-aware extraction is the right default.

**11/ (CTA)**
Takeaway: compression has been treated as a modeling problem. At the gateway it is a systems problem first.

The thing on the critical path of every request should be cheap before it is clever.

Paper + reproducible code below. Repost tweet 1 if it was useful.
[link]

---

### Hook variations (pick one for tweet 1)

1. *(used above)* "We replaced a 312 MB neural prompt compressor with NLP from the 1970s. At the LLM gateway it was 1213x faster AND more accurate."
2. "We benchmarked a neural prompt compressor at the gateway. It made requests slower. A 1970s algorithm beat it by 1213x and kept more answers."
3. "The best prompt compressor for a gateway holds zero model weights and runs in 0.3 ms. It also predates transformers. Real numbers and the terminal inside."
4. "At the LLM gateway, query-aware BM25 beat a neural prompt compressor on accuracy (96% vs 39%) and speed (1213x). Here is why that is not a fluke."

---

### Posting notes
- Best window: Tuesday to Thursday, 9 to 11am ET.
- Post all 11 at once as a native thread. Long technical threads lose reach when dripped.
- Keep the link out of tweet 1. Put it in tweet 11 and pin the paper link as the first reply.
- The debate hooks are "it made requests slower" (T4) and "query-aware vs query-blind" (T7). Lean into replies there in the first hour.
- Next-day quote-tweet: pull T9 (the honest caveat) on its own. Showing your own weak spots reads as credible and tends to travel.
