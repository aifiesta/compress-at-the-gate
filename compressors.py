"""
Extractive prompt compressors for the gateway/routing layer.

All classical methods are pure-CPU, dependency-light (numpy / scikit-learn /
networkx) and hold no model weights, so they can be co-located with a serving
process at ~zero additional accelerator footprint.

The neural baseline (SelectiveContextCompressor) is a faithful reimplementation
of token self-information pruning (Li et al., 2023, "Selective Context"): it
scores lexical units by the negative log-likelihood assigned by a small causal
LM and keeps the most "surprising" (information-dense) sentences. It needs a
transformer forward pass over the prompt, which is the cost we measure against.

Every compressor exposes:
    compress(context: str, question: str|None, keep_ratio: float) -> CompressionResult

`keep_ratio` is the target fraction of *sentences* to retain. Token-level
reduction is reported separately because sentences vary in length.
"""
from __future__ import annotations
import re
import math
from dataclasses import dataclass
from typing import Optional, List

import numpy as np


# --------------------------------------------------------------------------- #
# Sentence segmentation (lightweight, deterministic, dependency-free)
# --------------------------------------------------------------------------- #
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def split_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    # Primary split on sentence punctuation; fall back to newline / chunk.
    sents = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    if len(sents) <= 1:
        sents = [s.strip() for s in re.split(r"\n+", text) if s.strip()]
    return sents or [text]


_WORD = re.compile(r"[A-Za-z0-9]+")


def tokenize_words(text: str) -> List[str]:
    return _WORD.findall(text.lower())


@dataclass
class CompressionResult:
    text: str                 # compressed context
    kept_idx: List[int]       # indices of sentences kept (in original order)
    n_sentences: int          # number of sentences in the source


def _select_to_text(sentences: List[str], order: List[int], keep_ratio: float) -> CompressionResult:
    """Keep the top-scoring `keep_ratio` fraction of sentences, restore source order."""
    n = len(sentences)
    k = max(1, int(round(keep_ratio * n)))
    keep = sorted(order[:k])
    text = " ".join(sentences[i] for i in keep)
    return CompressionResult(text=text, kept_idx=keep, n_sentences=n)


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
class LeadCompressor:
    """Keep the first k sentences (a.k.a. the classic 'Lead-k' summary baseline)."""
    name = "Lead"

    def compress(self, context, question=None, keep_ratio=0.33) -> CompressionResult:
        sents = split_sentences(context)
        return _select_to_text(sents, list(range(len(sents))), keep_ratio)


class RandomCompressor:
    """Drop sentences uniformly at random (seeded). Lower-bound reference."""
    name = "Random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def compress(self, context, question=None, keep_ratio=0.33) -> CompressionResult:
        sents = split_sentences(context)
        order = list(self.rng.permutation(len(sents)))
        return _select_to_text(sents, order, keep_ratio)


# --------------------------------------------------------------------------- #
# Classical salience methods
# --------------------------------------------------------------------------- #
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx


def _tfidf_matrix(sentences: List[str]):
    vec = TfidfVectorizer(lowercase=True, stop_words="english", token_pattern=r"[A-Za-z0-9]+")
    try:
        X = vec.fit_transform(sentences)
    except ValueError:
        # all-stopword / empty corpus: fall back to raw counts
        vec = TfidfVectorizer(lowercase=True, token_pattern=r"[A-Za-z0-9]+")
        X = vec.fit_transform(sentences)
    return X, vec


class TfidfCompressor:
    """
    Query-agnostic salience: score each sentence by mean TF-IDF weight of its
    terms, with a mild lead-position prior. Captures globally salient sentences.
    """
    name = "TF-IDF"

    def __init__(self, position_prior: float = 0.15, query_aware: bool = False):
        self.position_prior = position_prior
        self.query_aware = query_aware

    def compress(self, context, question=None, keep_ratio=0.33) -> CompressionResult:
        sents = split_sentences(context)
        n = len(sents)
        if n <= 1:
            return _select_to_text(sents, list(range(n)), keep_ratio)
        X, vec = _tfidf_matrix(sents)
        # salience = mean nonzero tf-idf per sentence (row L2 already normalized,
        # so use row sum as a density proxy)
        salience = np.asarray(X.sum(axis=1)).ravel()
        salience = salience / (salience.max() + 1e-9)
        # position prior: earlier sentences get a small bonus
        pos = 1.0 - np.arange(n) / max(1, n - 1)
        score = salience + self.position_prior * pos
        if self.query_aware and question:
            q = vec.transform([question])
            qsim = cosine_similarity(X, q).ravel()
            qsim = qsim / (qsim.max() + 1e-9)
            score = score + qsim  # query relevance dominates when present
        order = list(np.argsort(-score))
        return _select_to_text(sents, order, keep_ratio)


class TextRankCompressor:
    """
    Graph centrality (Mihalcea & Tarau, 2004): PageRank over a sentence-similarity
    graph built from TF-IDF cosine. Optionally biased toward the query via a
    personalization vector (query-aware TextRank).
    """
    name = "TextRank"

    def __init__(self, query_aware: bool = False, sim_threshold: float = 0.0):
        self.query_aware = query_aware
        self.sim_threshold = sim_threshold

    def compress(self, context, question=None, keep_ratio=0.33) -> CompressionResult:
        sents = split_sentences(context)
        n = len(sents)
        if n <= 2:
            return _select_to_text(sents, list(range(n)), keep_ratio)
        X, vec = _tfidf_matrix(sents)
        S = cosine_similarity(X)
        np.fill_diagonal(S, 0.0)
        if self.sim_threshold > 0:
            S[S < self.sim_threshold] = 0.0
        G = nx.from_numpy_array(S)
        pers = None
        if self.query_aware and question:
            q = vec.transform([question])
            qsim = cosine_similarity(X, q).ravel() + 1e-6
            pers = {i: float(w) for i, w in enumerate(qsim / qsim.sum())}
        try:
            pr = nx.pagerank(G, alpha=0.85, personalization=pers, max_iter=200)
        except nx.PowerIterationFailedConvergence:
            deg = S.sum(axis=1)
            pr = {i: float(deg[i]) for i in range(n)}
        order = sorted(range(n), key=lambda i: -pr.get(i, 0.0))
        return _select_to_text(sents, order, keep_ratio)


class CentroidMMRCompressor:
    """
    Centroid relevance + Maximal Marginal Relevance (Carbonell & Goldstein, 1998).
    Greedily selects sentences that are close to the document/query centroid while
    being diverse w.r.t. already-selected sentences. Query-aware when a question
    is supplied.
    """
    name = "Centroid+MMR"

    def __init__(self, lam: float = 0.7, query_aware: bool = True):
        self.lam = lam
        self.query_aware = query_aware

    def compress(self, context, question=None, keep_ratio=0.33) -> CompressionResult:
        sents = split_sentences(context)
        n = len(sents)
        if n <= 1:
            return _select_to_text(sents, list(range(n)), keep_ratio)
        X, vec = _tfidf_matrix(sents)
        if self.query_aware and question:
            target = vec.transform([question])
        else:
            target = X.mean(axis=0)
            target = np.asarray(target)
        rel = cosine_similarity(X, target.reshape(1, -1) if hasattr(target, "reshape") else target).ravel()
        k = max(1, int(round(keep_ratio * n)))
        selected: List[int] = []
        candidates = list(range(n))
        sim_between = cosine_similarity(X)
        while candidates and len(selected) < k:
            if not selected:
                j = int(np.argmax(rel))
            else:
                best, bj = -1e9, candidates[0]
                for c in candidates:
                    div = max(sim_between[c][s] for s in selected)
                    mmr = self.lam * rel[c] - (1 - self.lam) * div
                    if mmr > best:
                        best, bj = mmr, c
                j = bj
            selected.append(j)
            candidates.remove(j)
        return _select_to_text(sents, selected + candidates, keep_ratio)


class BM25Compressor:
    """Okapi BM25 query relevance over sentences (Robertson & Zaragoza, 2009)."""
    name = "BM25"

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b

    def sentence_scores(self, context, question):
        """Return (sentences, per-sentence BM25 relevance to the question)."""
        sents = split_sentences(context)
        n = len(sents)
        if n <= 1 or not question:
            return sents, np.zeros(n)
        docs = [tokenize_words(s) for s in sents]
        q = tokenize_words(question)
        dl = np.array([len(d) for d in docs], dtype=float)
        avgdl = dl.mean() + 1e-9
        df = {}
        for d in docs:
            for t in set(d):
                df[t] = df.get(t, 0) + 1
        idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        scores = np.zeros(n)
        for i, d in enumerate(docs):
            tf = {}
            for t in d:
                tf[t] = tf.get(t, 0) + 1
            s = 0.0
            for t in q:
                if t in tf:
                    num = tf[t] * (self.k1 + 1)
                    den = tf[t] + self.k1 * (1 - self.b + self.b * dl[i] / avgdl)
                    s += idf.get(t, 0.0) * num / den
            scores[i] = s
        return sents, scores

    def compress(self, context, question=None, keep_ratio=0.33) -> CompressionResult:
        sents, scores = self.sentence_scores(context, question)
        n = len(sents)
        if n <= 1 or not question:
            return _select_to_text(sents, list(range(n)), keep_ratio)
        order = list(np.argsort(-scores))
        return _select_to_text(sents, order, keep_ratio)


# --------------------------------------------------------------------------- #
# Neural baseline: Selective Context (token self-information via a small LM)
# --------------------------------------------------------------------------- #
class SelectiveContextCompressor:
    """
    Faithful reimplementation of self-information pruning (Li et al., 2023).
    Scores each sentence by the mean token negative-log-likelihood (self-
    information) under a small causal LM, computed with a sliding window so
    arbitrarily long contexts are supported, then keeps the highest-information
    sentences. This is query-AGNOSTIC by construction (matching the original
    method) and requires a transformer forward pass over the whole prompt.
    """
    name = "SelectiveContext(neural)"

    def __init__(self, model_name: str = "distilgpt2", window: int = 512, device: str = "cpu"):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.torch = torch
        self.device = device
        self.window = window
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
        self.model_name = model_name

    def _token_nll(self, input_ids):
        torch = self.torch
        with torch.no_grad():
            out = self.model(input_ids)
            logits = out.logits[:, :-1, :]
            target = input_ids[:, 1:]
            logp = torch.log_softmax(logits, dim=-1)
            nll = -logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)  # [1, L-1]
        return nll[0].cpu().numpy()

    def sentence_scores(self, context):
        """Return (sentences, per-sentence mean self-information)."""
        torch = self.torch
        sents = split_sentences(context)
        n = len(sents)
        if n <= 1:
            return sents, np.zeros(n)
        ids_all, spans = [], []
        for s in sents:
            t = self.tok.encode(" " + s, add_special_tokens=False)
            if not t:
                t = self.tok.encode(".", add_special_tokens=False)
            spans.append((len(ids_all), len(ids_all) + len(t)))
            ids_all.extend(t)
        ids_all = np.array(ids_all, dtype=np.int64)
        L = len(ids_all)
        nll = np.zeros(L, dtype=np.float32)
        counted = np.zeros(L, dtype=bool)
        step = self.window
        for start in range(0, L, step):
            end = min(L, start + step)
            chunk = ids_all[start:end]
            if len(chunk) < 2:
                continue
            inp = torch.tensor(chunk[None, :], device=self.device)
            chunk_nll = self._token_nll(inp)
            nll[start + 1:start + len(chunk)] = chunk_nll
            counted[start + 1:start + len(chunk)] = True
        scores = np.zeros(n)
        for i, (a, b) in enumerate(spans):
            m = counted[a:b]
            scores[i] = nll[a:b][m].mean() if m.any() else 0.0
        return sents, scores

    def compress(self, context, question=None, keep_ratio=0.33) -> CompressionResult:
        sents, scores = self.sentence_scores(context)
        if len(sents) <= 1:
            return _select_to_text(sents, list(range(len(sents))), keep_ratio)
        order = list(np.argsort(-scores))
        return _select_to_text(sents, order, keep_ratio)


# Registry of the classical (production-deployable) methods
CLASSICAL = {
    "Lead": lambda: LeadCompressor(),
    "Random": lambda: RandomCompressor(seed=0),
    "TF-IDF": lambda: TfidfCompressor(query_aware=False),
    "TF-IDF+Q": lambda: TfidfCompressor(query_aware=True),
    "TextRank": lambda: TextRankCompressor(query_aware=False),
    "TextRank+Q": lambda: TextRankCompressor(query_aware=True),
    "Centroid+MMR": lambda: CentroidMMRCompressor(query_aware=True),
    "BM25+Q": lambda: BM25Compressor(),
}
