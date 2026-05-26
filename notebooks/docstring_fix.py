#!/usr/bin/env python3
"""Programmatic pass over docstrings only. Eliminates em dashes,
semicolons, and a fixed set of contractions inside every module,
class, and function docstring of the target .py files. Leaves all
non-docstring code untouched.

Usage: docstring_fix.py file.py [file.py ...]

The pass is deliberately conservative. Em dashes become periods (and
the following character is capitalised if the resulting sentence
begins lowercase). Semicolons become periods. Contractions expand to
their full form. Repeated phrases (3+ occurrences of any 5-gram) are
left alone, since fixing those needs prose rewriting rather than
regex.
"""

from __future__ import annotations

import ast
import re
import sys

CONTRACTIONS = {
    "don't": "do not", "Don't": "Do not",
    "doesn't": "does not", "Doesn't": "Does not",
    "isn't": "is not", "Isn't": "Is not",
    "aren't": "are not", "Aren't": "Are not",
    "wasn't": "was not", "Wasn't": "Was not",
    "weren't": "were not", "Weren't": "Were not",
    "won't": "will not", "Won't": "Will not",
    "wouldn't": "would not", "Wouldn't": "Would not",
    "shouldn't": "should not", "Shouldn't": "Should not",
    "couldn't": "could not", "Couldn't": "Could not",
    "hasn't": "has not", "Hasn't": "Has not",
    "haven't": "have not", "Haven't": "Have not",
    "hadn't": "had not", "Hadn't": "Had not",
    "can't": "cannot", "Can't": "Cannot",
    "we'll": "we will", "We'll": "We will",
    "we're": "we are", "We're": "We are",
    "we've": "we have", "We've": "We have",
    "we'd": "we would", "We'd": "We would",
    "you'll": "you will", "You'll": "You will",
    "you're": "you are", "You're": "You are",
    "you've": "you have", "You've": "You have",
    "they're": "they are", "They're": "They are",
    "they've": "they have", "They've": "They have",
    "it's": "it is", "It's": "It is",
    "that's": "that is", "That's": "That is",
    "there's": "there is", "There's": "There is",
    "here's": "here is", "Here's": "Here is",
    "what's": "what is", "What's": "What is",
    "let's": "let us", "Let's": "Let us",
}


def fix_em_dashes(s: str) -> str:
    """``X — Y`` becomes ``X. Y``, with ``Y`` capitalised if it begins
    lowercase. Triple-hyphen ``---`` is treated identically."""
    def repl(m: re.Match) -> str:
        rest = m.group("rest")
        if rest and rest[0].islower():
            rest = rest[0].upper() + rest[1:]
        return f". {rest}" if rest else "."

    # ``foo — bar`` or ``foo---bar`` (the dash may be tight against either side)
    s = re.sub(r"\s*(?:—|---)\s*(?P<rest>\w)", repl, s)
    # leftover lonesome dashes
    s = s.replace("—", ".").replace("---", ".")
    return s


def fix_semicolons(s: str) -> str:
    """Replace semicolons with periods. The following word is
    capitalised if it begins lowercase."""
    def repl(m: re.Match) -> str:
        rest = m.group("rest")
        if rest and rest[0].islower():
            rest = rest[0].upper() + rest[1:]
        return f". {rest}" if rest else "."

    s = re.sub(r"\s*;\s*(?P<rest>\w)", repl, s)
    return s.replace(";", ".")


def fix_contractions(s: str) -> str:
    for k, v in CONTRACTIONS.items():
        s = re.sub(rf"\b{re.escape(k)}\b", v, s)
    return s


def transform(doc: str) -> str:
    doc = fix_em_dashes(doc)
    doc = fix_semicolons(doc)
    doc = fix_contractions(doc)
    return doc


def process(path: str) -> tuple[int, int]:
    src = open(path).read()
    tree = ast.parse(src)

    # Collect docstring constant nodes (Expr -> Constant[str]).
    targets: list[ast.Constant] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                targets.append(body[0].value)

    # Sort in reverse source order so earlier replacements do not shift
    # later line/col positions.
    targets.sort(key=lambda c: (c.lineno, c.col_offset), reverse=True)

    lines = src.splitlines(keepends=True)
    changes = 0
    total = len(targets)

    for const in targets:
        start_line = const.lineno - 1
        end_line = const.end_lineno - 1
        start_col = const.col_offset
        end_col = const.end_col_offset
        # Pull out the literal text (the """...""" or '''...''' form).
        if start_line == end_line:
            original = lines[start_line][start_col:end_col]
        else:
            original = (
                lines[start_line][start_col:]
                + "".join(lines[start_line + 1 : end_line])
                + lines[end_line][:end_col]
            )

        # Identify the opening / closing quotes.
        match = re.match(r'^([rbRBuU]*)(\'{3}|"{3}|\'|")', original)
        if not match:
            continue
        prefix, quote = match.group(1), match.group(2)
        if len(quote) != 3:
            continue  # not a triple-quoted docstring; skip
        inner_start = len(prefix) + len(quote)
        inner_end = -len(quote)
        inner = original[inner_start:inner_end]
        new_inner = transform(inner)
        if new_inner == inner:
            continue
        new_literal = original[:inner_start] + new_inner + original[inner_end:]

        # Splice the new literal back in.
        if start_line == end_line:
            lines[start_line] = (
                lines[start_line][:start_col]
                + new_literal
                + lines[start_line][end_col:]
            )
        else:
            head = lines[start_line][:start_col] + new_literal
            # Merge head with trailing portion of end_line.
            tail = lines[end_line][end_col:]
            merged = head + tail
            new_lines = merged.splitlines(keepends=True)
            lines[start_line : end_line + 1] = new_lines
        changes += 1

    if changes:
        open(path, "w").write("".join(lines))

    return changes, total


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: docstring_fix.py file.py [...]", file=sys.stderr)
        return 1
    for arg in sys.argv[1:]:
        try:
            changed, total = process(arg)
            print(f"{arg}: {changed}/{total} docstrings updated")
        except Exception as e:
            print(f"{arg}: FAILED {type(e).__name__}: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
