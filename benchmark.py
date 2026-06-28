"""
Multi-document "lost-in-the-middle" QA benchmark built from SQuAD v1.1.

Each item concatenates one gold paragraph (which contains the answer) with
`n_distractors` distractor paragraphs drawn from unrelated articles, mimicking
the multi-document RAG setting used by LongLLMLingua / "Lost in the Middle".
The gold paragraph is placed at a controlled position so we can also study
position bias.

Primary quality metric: ANSWER RETENTION -- does the compressed context still
contain the gold answer span? This is an intrinsic, fully reproducible measure
of extractive-compression quality that needs no downstream LLM call. We also
report gold-sentence recall and token-reduction.
"""
from __future__ import annotations
import re
import string
import random
from dataclasses import dataclass, field
from typing import List, Optional

import tiktoken

from compressors import split_sentences

_ENC = tiktoken.get_encoding("cl100k_base")


def n_tokens(text: str) -> int:
    return len(_ENC.encode(text))


# ---- SQuAD-style answer normalization (official) --------------------------- #
def normalize_answer(s: str) -> str:
    def remove_articles(t):
        return re.sub(r"\b(a|an|the)\b", " ", t)

    def white_space_fix(t):
        return " ".join(t.split())

    def remove_punc(t):
        return "".join(ch for ch in t if ch not in set(string.punctuation))

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def answer_present(text: str, answer: str) -> bool:
    return normalize_answer(answer) in normalize_answer(text)


@dataclass
class QAItem:
    question: str
    answer: str
    context: str          # full multi-document context
    gold_paragraph: str
    gold_sentence: str
    gold_position: int    # index of gold paragraph among the passages
    n_passages: int
    src_tokens: int


def build_multidoc_squad(n_items: int = 200, n_distractors: int = 7,
                         seed: int = 13, gold_position: str = "middle") -> List[QAItem]:
    """
    gold_position: 'first' | 'middle' | 'last' | 'random'
    """
    from datasets import load_dataset
    ds = load_dataset("squad", split="validation")
    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)

    items: List[QAItem] = []
    pool = idxs[:]  # for distractors
    used = 0
    for i in idxs:
        if len(items) >= n_items:
            break
        ex = ds[i]
        ans_list = ex["answers"]["text"]
        if not ans_list:
            continue
        answer = ans_list[0]
        gold_para = ex["context"].strip()
        if not answer_present(gold_para, answer):
            continue
        # locate gold sentence
        gold_sentence = ""
        for s in split_sentences(gold_para):
            if answer_present(s, answer):
                gold_sentence = s
                break
        if not gold_sentence:
            continue
        # sample distractors that do NOT contain the answer and differ in title
        distractors = []
        tries = 0
        while len(distractors) < n_distractors and tries < 200:
            tries += 1
            j = rng.choice(pool)
            if j == i:
                continue
            dpar = ds[j]["context"].strip()
            if ds[j]["title"] == ex["title"]:
                continue
            if answer_present(dpar, answer):
                continue
            distractors.append(dpar)
        if len(distractors) < n_distractors:
            continue
        # place gold
        passages = distractors[:]
        if gold_position == "first":
            pos = 0
        elif gold_position == "last":
            pos = len(passages)
        elif gold_position == "random":
            pos = rng.randint(0, len(passages))
        else:  # middle
            pos = len(passages) // 2
        passages.insert(pos, gold_para)
        context = "\n\n".join(passages)
        items.append(QAItem(
            question=ex["question"].strip(),
            answer=answer,
            context=context,
            gold_paragraph=gold_para,
            gold_sentence=gold_sentence,
            gold_position=pos,
            n_passages=len(passages),
            src_tokens=n_tokens(context),
        ))
        used += 1
    return items


# --------------------------------------------------------------------------- #
# Quality evaluation
# --------------------------------------------------------------------------- #
@dataclass
class QualityStats:
    method: str
    keep_ratio: float
    n: int
    answer_retention: float          # fraction of items where answer survives
    gold_sentence_recall: float      # fraction where the gold sentence survives
    mean_token_reduction: float      # 1 - kept_tokens/src_tokens
    mean_kept_tokens: float
    mean_src_tokens: float


def evaluate_quality(method, items: List[QAItem], keep_ratio: float) -> QualityStats:
    ret = 0
    grec = 0
    red = []
    kept_toks = []
    src_toks = []
    for it in items:
        res = method.compress(it.context, it.question, keep_ratio=keep_ratio)
        comp = res.text
        if answer_present(comp, it.answer):
            ret += 1
        # gold sentence recall: is the (normalized) gold sentence a substring?
        if normalize_answer(it.gold_sentence)[:120] in normalize_answer(comp):
            grec += 1
        kt = n_tokens(comp)
        kept_toks.append(kt)
        src_toks.append(it.src_tokens)
        red.append(1 - kt / max(1, it.src_tokens))
    n = len(items)
    import numpy as np
    return QualityStats(
        method=getattr(method, "name", method.__class__.__name__),
        keep_ratio=keep_ratio,
        n=n,
        answer_retention=ret / n,
        gold_sentence_recall=grec / n,
        mean_token_reduction=float(np.mean(red)),
        mean_kept_tokens=float(np.mean(kept_toks)),
        mean_src_tokens=float(np.mean(src_toks)),
    )
