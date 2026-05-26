#!/usr/bin/env python3
"""Style match score for notebook markdown vs the author's C5 voice profile.

Ported from ~/Dev/typesetting/phd-thesis/style_score.py. The targets and
weights below are calibrated against the LaTeX source of the author's TPLP
journal submission and represent the author's established academic voice.

Usage:
    python3 style_score.py notebook.ipynb [notebook2.ipynb ...]

Outputs ``style_score.json`` with per-metric scores and the composite. A
composite >= 0.85 hits the same target the phd-thesis autoresearch loop uses.
"""

from __future__ import annotations

import ast
import json
import math
import re
import sys
from pathlib import Path


def extract_py(path: str) -> str:
    """Concatenate every docstring (module, class, function, async function)
    in a ``.py`` source file. Reading order matches the AST walk so callers
    see the file in declaration order."""
    tree = ast.parse(open(path).read())
    pieces: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node, clean=True)
            if doc:
                pieces.append(doc)
    return "\n\n".join(pieces)


def extract_md(path: str) -> str:
    """Return prose from a notebook (``.ipynb``), plain markdown (``.md``),
    or Python source (``.py``, docstrings only). Strips inline code, math,
    links, and fenced code blocks so the metrics only see prose."""
    if path.endswith(".py"):
        text = extract_py(path)
    elif path.endswith(".ipynb"):
        nb = json.load(open(path))
        pieces: list[str] = []
        for cell in nb.get("cells", []):
            if cell.get("cell_type") != "markdown":
                continue
            src = cell.get("source", "")
            if isinstance(src, list):
                src = "".join(src)
            pieces.append(src)
        text = "\n\n".join(pieces)
    else:
        text = open(path).read()
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)  # fenced code blocks
    text = re.sub(r":[a-zA-Z]+:`+[^`]+`+", "", text)          # sphinx roles
    text = re.sub(r"`+[^`]+`+", "", text)                     # inline / RST code
    text = re.sub(r"\$[^$]+\$", "", text)                     # math
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)      # link → label
    text = re.sub(r"^#+\s.*$", "", text, flags=re.MULTILINE)  # entire heading lines
    text = re.sub(r"^[\s|:\-]+$", "", text, flags=re.MULTILINE)  # md table separators
    return text


def words(t: str) -> list[str]:
    return re.findall(r"\b[a-zA-Z]+\b", t.lower())


def sentences(t: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", t) if len(s.strip()) > 10]


def paragraphs(t: str) -> list[str]:
    ps = re.split(r"\n\s*\n", t)
    return [p.strip() for p in ps if len(p.strip()) > 30 and not p.strip().startswith("=")]


def syllables(w: str) -> int:
    w = w.lower()
    c = 0
    v = "aeiouy"
    if w[0] in v:
        c += 1
    for i in range(1, len(w)):
        if w[i] in v and w[i - 1] not in v:
            c += 1
    if w.endswith("e"):
        c -= 1
    return max(c, 1)


def count_patterns(text: str, patterns: list[str]) -> int:
    t = text.lower()
    return sum(len(re.findall(p, t)) for p in patterns)


def gaussian(val: float, target: float, sigma: float) -> float:
    return math.exp(-((val - target) ** 2) / (2 * sigma**2))


# Calibrated against C5 (TPLP journal submission) LaTeX source.
SURFACE_METRICS = [
    # (name,              sigma, weight, target)
    ("mean_para_len",      15.0, 1.0,  49.0),
    ("we_per_k",            1.5, 2.0,   6.0),
    ("our_per_k",           0.8, 1.0,   1.5),
    ("passive_per_k",       1.0, 1.5,   5.2),
    ("the_opener_%",        3.0, 1.5,  13.0),
    ("we_opener_%",         2.0, 1.5,   6.0),
    ("flesch_kincaid",      1.5, 1.0,  11.5),
    ("sent_std",            3.0, 0.5,  17.0),
    ("em_dash_per_k",       0.3, 5.0,   0.0),
]

RHETORIC_METRICS = [
    ("hedge_per_k",         0.5, 2.0,   1.5),
    ("hedge_claim_ratio",   0.3, 2.0,   1.0),
    ("evaluative_per_k",    1.0, 1.0,   1.2),
]

FINGERPRINT_METRICS = [
    ("semicolon_per_k",     0.3, 5.0,   0.0),
    ("formal_register_per_k", 0.3, 1.0, 0.5),
    ("long_paren_per_k",    1.0, 1.0,   0.7),
    ("contractions",        0.5, 3.0,   0.0),
    ("that_per_k",          5.0, 0.5,  19.3),
    ("repeated_phrases",    3.0, 3.0,   0.0),
]

HEDGE_PATTERNS = [
    r"\bwe posit\b", r"\bwe argue\b", r"\bwe believe\b", r"\bwe conjecture\b",
    r"\bcould\b", r"\bmight\b", r"\bpotentially\b", r"\bpossibly\b",
    r"\bwe expect\b", r"\bit is possible\b", r"\bto some extent\b",
    r"\bperhaps\b", r"\bmay\b(?!\.)",
    r"\barguably\b", r"\bapparently\b", r"\bplausibly\b",
]

CLAIM_PATTERNS = [
    r"\bwe show\b", r"\bwe demonstrate\b", r"\bwe prove\b",
    r"\bour contribution\b", r"\bour approach\b", r"\bour key\b",
    r"\bthe first\b", r"\bthe only\b", r"\bunlike\b",
    r"\bin contrast\b", r"\bmore critically\b", r"\bmore importantly\b",
]

EVALUATIVE_PATTERNS = [
    r"\bprohibitively\b", r"\bfundamentally\b", r"\bsignificantly\b",
    r"\bunusable\b", r"\bdramatic\b", r"\bcrucial\b", r"\bmerely\b",
    r"\bprecisely\b", r"\bexactly\b", r"\bentirely\b", r"\bcritically\b",
    r"\bimpractical\b", r"\binherently\b", r"\bnaturally\b",
    r"\btrivially\b", r"\bseverely\b", r"\bkey insight\b",
    r"\bkey innovation\b", r"\bcentral\b",
]

FORMAL_PATTERNS = [
    r"\bsans\b", r"\bhence\b", r"\bnonetheless\b", r"\bthereof\b",
    r"\btherein\b", r"\bwhereby\b", r"\bwherein\b", r"\baforementioned\b",
    r"\bhitherto\b", r"\binsofar\b", r"\bnotwithstanding\b", r"\bwhilst\b",
    r"\bin a similar vein\b", r"\bis akin to\b", r"\blet alone\b",
    r"\bin spite of\b",
]


def compute(path: str, verbose: bool = False) -> tuple[float, dict]:
    text = extract_md(path)
    ws = words(text)
    ss = sentences(text)
    ps = paragraphs(text)
    n = len(ws)
    if n == 0 or not ss:
        return 0.0, {"error": "no prose", "path": path}

    para_lens = [len(words(p)) for p in ps]
    mean_para = sum(para_lens) / len(para_lens) if para_lens else 0
    sent_lens = [len(s.split()) for s in ss]
    sent_mean = sum(sent_lens) / len(sent_lens)
    sent_std = (sum((x - sent_mean) ** 2 for x in sent_lens) / len(sent_lens)) ** 0.5

    openers = [s.split()[0].lower().strip("*_") for s in ss if s.split()]
    the_pct = sum(1 for o in openers if o == "the") / len(openers) * 100
    we_opener = sum(1 for o in openers if o == "we") / len(openers) * 100

    we_k = len(re.findall(r"\bwe\b", text.lower())) / n * 1000
    our_k = len(re.findall(r"\bour\b", text.lower())) / n * 1000
    passive_k = (
        len(re.findall(r"\b(?:is|are|was|were|been|being)\s+\w+ed\b", text.lower()))
        / n * 1000
    )
    em_k = (text.count("---") + text.count("—")) / n * 1000
    fk = 0.39 * (n / len(ss)) + 11.8 * (sum(syllables(w) for w in ws) / n) - 15.59

    hedge_count = count_patterns(text, HEDGE_PATTERNS)
    claim_count = count_patterns(text, CLAIM_PATTERNS)
    eval_count = count_patterns(text, EVALUATIVE_PATTERNS)

    hedge_k = hedge_count / n * 1000
    eval_k = eval_count / n * 1000
    hedge_claim = hedge_count / claim_count if claim_count > 0 else 0.0

    semicolon_k = text.count(";") / n * 1000
    formal_k = count_patterns(text, FORMAL_PATTERNS) / n * 1000

    parens = re.findall(r"\([^)]+\)", text)
    long_parens = [p for p in parens if len(p.split()) > 5]
    long_paren_k = len(long_parens) / n * 1000

    contractions = len(
        re.findall(r"\b\w+'t\b|\b\w+'re\b|\b\w+'ve\b|\b\w+'ll\b", text.lower())
    )

    that_k = len(re.findall(r"\bthat\b", text.lower())) / n * 1000

    prose_words = re.findall(r"\b[a-z]+\b", text.lower())
    fivegrams: dict[str, int] = {}
    for i in range(len(prose_words) - 4):
        gram = " ".join(prose_words[i : i + 5])
        fivegrams[gram] = fivegrams.get(gram, 0) + 1
    repeated_phrases = sum(1 for _, c in fivegrams.items() if c >= 3)

    vals = {
        "mean_para_len": mean_para,
        "we_per_k": we_k,
        "our_per_k": our_k,
        "passive_per_k": passive_k,
        "the_opener_%": the_pct,
        "we_opener_%": we_opener,
        "flesch_kincaid": fk,
        "sent_std": sent_std,
        "em_dash_per_k": em_k,
        "hedge_per_k": hedge_k,
        "hedge_claim_ratio": hedge_claim,
        "evaluative_per_k": eval_k,
        "semicolon_per_k": semicolon_k,
        "formal_register_per_k": formal_k,
        "long_paren_per_k": long_paren_k,
        "contractions": contractions,
        "that_per_k": that_k,
        "repeated_phrases": repeated_phrases,
    }

    all_metrics = SURFACE_METRICS + RHETORIC_METRICS + FINGERPRINT_METRICS

    total_w = 0.0
    total_ws = 0.0
    results: dict[str, dict] = {}
    worst: list[tuple[str, float, float, float]] = []

    if verbose:
        print(f"\n  {'Metric':22s} {'Value':>7} {'Target':>7} {'σ':>5} {'Score':>6} {'Wt':>4}")
        print(f"  {'-' * 62}")

    for name, sigma, weight, target in all_metrics:
        val = vals[name]
        s = gaussian(val, target, sigma)
        total_ws += s * weight
        total_w += weight
        results[name] = {"value": round(val, 2), "target": target, "score": round(s, 3)}
        if s < 0.5:
            worst.append((name, val, target, s))
        if verbose:
            flag = " !" if s < 0.5 else "  " if s < 0.8 else ""
            print(f"  {name:22s} {val:>7.2f} {target:>7.1f} {sigma:>5.1f} {s:>6.3f} {weight:>4.1f}{flag}")

    composite = total_ws / total_w

    if verbose:
        print(f"\n  STYLE SCORE: {composite:.3f}")
        print(f"  n_words={n}  n_sentences={len(ss)}  n_paragraphs={len(ps)}")
        if worst:
            print("  WORST: " + ", ".join(f"{n}={v:.2f} (target {t})" for n, v, t, _ in worst))

    output: dict = {
        "path": path,
        "composite_score": round(composite, 3),
        "n_words": n,
        "n_sentences": len(ss),
        "n_paragraphs": len(ps),
        "metrics": results,
    }
    if worst:
        output["worst_metrics"] = [m[0] for m in worst]

    return composite, output


def main() -> int:
    if len(sys.argv) < 2:
        print(
            "usage: style_score.py <notebook.ipynb | doc.md | source.py> [...]",
            file=sys.stderr,
        )
        return 1

    all_results: list[dict] = []
    for arg in sys.argv[1:]:
        path = Path(arg)
        if not path.exists():
            print(f"missing: {path}", file=sys.stderr)
            continue
        print(f"\n=== {path.name} ===")
        _, result = compute(str(path), verbose=True)
        all_results.append(result)

    Path("style_score.json").write_text(json.dumps(all_results, indent=2) + "\n")
    print(f"\n→ style_score.json ({len(all_results)} notebook{'s' if len(all_results) != 1 else ''})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
