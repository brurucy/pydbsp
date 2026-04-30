"""Bilinear operators — sort-merge join over indexed-zset streams."""

from collections.abc import Callable
from itertools import product as cartesian_product

from pydbsp.core import DBSPTime
from pydbsp.indexed_zset import IndexedZSet, sort_merge_keys
from pydbsp.stream import Lift2, Stream
from pydbsp.stream.operators.linear import (
    Delay,
    Integrate,
    StreamAddition,
)
from pydbsp.zset import ZSet, ZSetAddition


def sort_merge_join[I, A, B, C](
    left: IndexedZSet[I, A],
    right: IndexedZSet[I, B],
    proj: Callable[[I, A, B], C | None],
) -> ZSet[C]:
    """Equi-join via sort-merge over the shared index. ``proj`` may
    return ``None`` to drop a pair.

    **Invariant:** ``index_to_value[k]`` only contains values whose
    weight in ``inner`` is nonzero. Producers (IndexedZSetAddition)
    must maintain this — we no longer guard per-pair.
    """
    out: dict[C, int] = {}
    l_inner = left.inner
    r_inner = right.inner
    for key in sort_merge_keys(left.index, right.index):
        # Spine is append-only; a cancellation in ``IndexedZSetAddition.add``
        # removes a key from ``index_to_value`` but leaves it in ``index``.
        # Use ``.get`` and skip orphan keys.
        l_bucket = left.index_to_value.get(key)
        r_bucket = right.index_to_value.get(key)
        if not l_bucket or not r_bucket:
            continue
        for l_val, r_val in cartesian_product(l_bucket, r_bucket):
            c = proj(key, l_val, r_val)
            if c is None:
                continue
            out[c] = out.get(c, 0) + l_inner[l_val] * r_inner[r_val]
    return ZSet({k: v for k, v in out.items() if v != 0})


def DLDSortMergeJoin[I, A, B, C, T: tuple[int, ...]](
    diff_a: Stream[IndexedZSet[I, A], T],
    diff_b: Stream[IndexedZSet[I, B], T],
    proj: Callable[[I, A, B], C],
    out_group: ZSetAddition[C],
    lattice: DBSPTime[T],
    outer_axis: int = 0,
    inner_axis: int = 1,
) -> Stream[ZSet[C], T]:
    """4-term doubly-incremental sort-merge join on a flat product
    lattice, parameterised by ``outer_axis`` / ``inner_axis``.
    """
    sg: StreamAddition[ZSet[C], T] = StreamAddition(out_group, lattice)

    int_a_o = Integrate(diff_a, lattice, axis=outer_axis)
    del_int_a_o = Delay(int_a_o, lattice, axis=outer_axis)
    int_b_o = Integrate(diff_b, lattice, axis=outer_axis)
    del_int_b_o = Delay(int_b_o, lattice, axis=outer_axis)

    int_a_i = Integrate(diff_a, lattice, axis=inner_axis)
    int_b_i = Integrate(diff_b, lattice, axis=inner_axis)
    int_a_oi = Integrate(int_a_i, lattice, axis=outer_axis)
    int_b_oi = Integrate(int_b_i, lattice, axis=outer_axis)

    del_int_b_i = Delay(int_b_i, lattice, axis=inner_axis)
    del_int_b_oi = Delay(int_b_oi, lattice, axis=inner_axis)

    term: Callable[[IndexedZSet[I, A], IndexedZSet[I, B]], ZSet[C]] = (
        lambda la, rb: sort_merge_join(la, rb, proj)
    )

    j1 = Lift2(del_int_a_o, del_int_b_i, term, out_group)
    j2 = Lift2(int_a_oi, diff_b, term, out_group)
    j3 = Lift2(int_a_i, del_int_b_o, term, out_group)
    j4 = Lift2(diff_a, del_int_b_oi, term, out_group)

    return sg.add(sg.add(j1, j2), sg.add(j3, j4))


def DLDSortMergeJoin3D[I, A, B, C, T: tuple[int, ...]](
    diff_a: Stream[IndexedZSet[I, A], T],
    diff_b: Stream[IndexedZSet[I, B], T],
    proj: Callable[[I, A, B], C],
    out_group: ZSetAddition[C],
    lattice: DBSPTime[T],
    axes: tuple[int, int, int] = (0, 1, 2),
) -> Stream[ZSet[C], T]:
    """8-term triply-incremental sort-merge join on a 3-axis flat
    product lattice. ``axes`` names the three active axes in order
    ``(α₀, α₁, α₂)``; integration/delay operators wind along these.

    Derived from the identity
    ``Δ_{α₀}Δ_{α₁}Δ_{α₂} B(I_{α₀}I_{α₁}I_{α₂} a, I_{α₀}I_{α₁}I_{α₂} b)``.
    Applying ``Δ_α B(A,A) = B(A, Δ_α A) + B(Δ_α A, z⁻¹_α A)`` once per
    axis expands to 8 bilinear products, indexed by subsets
    ``S ⊆ {α₀, α₁, α₂}``:

        term_S = B( ∏_{α∈S} I_α · a ,  ∏_{α∉S} z⁻¹_α I_α · b )

    ``S = {α₀,α₁,α₂}`` gives ``B(I_0 I_1 I_2 a, b)``; ``S = ∅`` gives
    ``B(a, z⁻¹_0 z⁻¹_1 z⁻¹_2 I_0 I_1 I_2 b)``. For 2D lattices use
    ``DLDSortMergeJoin`` (4 terms); the pattern extends to N axes with
    ``2^N`` terms.
    """
    sg: StreamAddition[ZSet[C], T] = StreamAddition(out_group, lattice)

    def integ(s, ax_set):
        for ax in ax_set:
            s = Integrate(s, lattice, axis=ax)
        return s

    def delay_integ(s, ax_set):
        s = integ(s, ax_set)
        for ax in ax_set:
            s = Delay(s, lattice, axis=ax)
        return s

    term: Callable[[IndexedZSet[I, A], IndexedZSet[I, B]], ZSet[C]] = (
        lambda la, rb: sort_merge_join(la, rb, proj)
    )

    result: Stream[ZSet[C], T] | None = None
    for mask in range(1 << 3):
        S = [axes[k] for k in range(3) if (mask >> k) & 1]
        S_c = [axes[k] for k in range(3) if not ((mask >> k) & 1)]
        left = integ(diff_a, S)
        right = delay_integ(diff_b, S_c)
        tt = Lift2(left, right, term, out_group)
        result = tt if result is None else sg.add(result, tt)
    assert result is not None
    return result


def DLDSortMergeJoin3DStaticInnerRight[I, A, B, C, T: tuple[int, ...]](
    diff_a: Stream[IndexedZSet[I, A], T],
    diff_b: Stream[IndexedZSet[I, B], T],
    proj: Callable[[I, A, B], C],
    out_group: ZSetAddition[C],
    lattice: DBSPTime[T],
    axes: tuple[int, int, int] = (0, 1, 2),
) -> Stream[ZSet[C], T]:
    """Specialized 3D join when the right operand is static on the
    innermost axis ``axes[2]``.

    Group the 8 subset terms of ``DLDSortMergeJoin3D`` by whether the
    inner axis is integrated on the left or delayed-integrated on the
    right:

    * ``α₂ ∈ S``  ->  ``DLDSortMergeJoin(I_{α₂} a, b)`` over ``(α₀, α₁)``
    * ``α₂ ∉ S``  ->  ``DLDSortMergeJoin(a, z⁻¹_{α₂} I_{α₂} b)`` over
      ``(α₀, α₁)``

    This preserves the full doubly-incremental structure on the outer
    and stratum axes while eliminating the unnecessary expansion of the
    right operand along the inner axis.
    """
    outer_axis, stratum_axis, inner_axis = axes
    sg: StreamAddition[ZSet[C], T] = StreamAddition(out_group, lattice)

    left_inner_full = Integrate(diff_a, lattice, axis=inner_axis)
    right_inner_tail = Delay(Integrate(diff_b, lattice, axis=inner_axis), lattice, axis=inner_axis)

    term_left_inner = DLDSortMergeJoin(
        left_inner_full,
        diff_b,
        proj,
        out_group,
        lattice,
        outer_axis=outer_axis,
        inner_axis=stratum_axis,
    )
    term_right_inner = DLDSortMergeJoin(
        diff_a,
        right_inner_tail,
        proj,
        out_group,
        lattice,
        outer_axis=outer_axis,
        inner_axis=stratum_axis,
    )
    return sg.add(term_left_inner, term_right_inner)
