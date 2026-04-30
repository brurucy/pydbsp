from pydbsp.zset import ZSet


def H[T](i: ZSet[T], d: ZSet[T]) -> ZSet[T]:
    """Threshold-crossing indicator from Lean relational_incremental.lean:12.

    ``H(i, d)(a)``:

    * ``-1`` if ``i(a) > 0`` and ``(i+d)(a) ≤ 0``   (element leaves the distinct set)
    * ``+1`` if ``i(a) ≤ 0`` and ``(i+d)(a) > 0``   (element enters the distinct set)
    *  ``0`` otherwise

    Only elements in ``d`` can cross the threshold — for ``a`` absent
    from ``d``, ``new == old`` and neither condition fires. Iterating
    ``d`` alone is O(|d|) instead of O(|i|+|d|); huge win when ``i``
    is a large cumulative state and ``d`` is a small delta.
    """
    i_inner = i.inner
    out: dict[T, int] = {}
    for a, delta in d.inner.items():
        old = i_inner.get(a, 0)
        new = old + delta
        if old > 0:
            if new <= 0:
                out[a] = -1
        elif new > 0:
            out[a] = 1
    return ZSet(out)
