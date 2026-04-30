"""Sort-merge-indexed variant of ``IncrementalRDFSMaterialization``
on the flat 2-D product lattice.

All six RDFS-rule joins are keyed on a single integer field of the
``RDFTuple`` — natural sort-merge candidates. Topology matches
``rdfs.py``; only the joins differ.
"""

from pydbsp.core import Time1D, Time2D
from pydbsp.indexed_zset.operators.bilinear import DLDSortMergeJoin
from pydbsp.indexed_zset.operators.linear import Index
from pydbsp.evaluator import Evaluator
from pydbsp.stream import Input
from pydbsp.stream.functions.linear import (
    TimeAxisElimination,
    TimeAxisIntroduction,
)
from pydbsp.stream.operators.linear import Delay
from pydbsp.stream.zset.operators.binary import DLDDistinct
from pydbsp.stream.zset.operators.linear import Select
from pydbsp.zset import ZSetAddition

from pydbsp.algorithms.rdfs import (
    DOMAIN,
    RANGE,
    RDFSCircuit,
    RDFTuple,
    SCO,
    SPO,
    TYPE,
)


Time = tuple[int, int]


def IncrementalRDFSMaterializationWithIndexing(parallelism: int = 1) -> RDFSCircuit:
    """Builds an RDFS materialization circuit with sort-merge-join
    indexing. Push ``abox`` (data) and ``tbox`` (schema) tuples at outer
    ticks, call ``saturate_rdfs``, read the materialization via
    ``observable_at``. Faster than ``IncrementalRDFSMaterialization`` on
    larger graphs.
    """
    tuple_group: ZSetAddition[RDFTuple] = ZSetAddition()
    lattice = Time2D

    abox = Input(tuple_group, Time1D)
    tbox = Input(tuple_group, Time1D)

    abox_2d = TimeAxisIntroduction(abox, tuple_group, lattice, axis=1)
    tbox_2d = TimeAxisIntroduction(tbox, tuple_group, lattice, axis=1)

    tbox_domain = Select(tbox_2d, lambda t: t[1] == DOMAIN)
    tbox_range = Select(tbox_2d, lambda t: t[1] == RANGE)

    state = Input(tuple_group, lattice)
    delay_state = Delay(state, lattice, axis=1)

    state_sco = Select(delay_state, lambda t: t[1] == SCO)
    state_spo = Select(delay_state, lambda t: t[1] == SPO)
    state_type = Select(delay_state, lambda t: t[1] == TYPE)
    state_non_type = Select(
        delay_state,
        lambda t: t[1] not in (SCO, SPO, TYPE, DOMAIN, RANGE),
    )

    # Indexed sides of the six rules.
    sco_by_dst = Index(state_sco, lambda t: t[2], tuple_group)
    sco_by_src = Index(state_sco, lambda t: t[0], tuple_group)
    spo_by_dst = Index(state_spo, lambda t: t[2], tuple_group)
    spo_by_src = Index(state_spo, lambda t: t[0], tuple_group)
    type_by_obj = Index(state_type, lambda t: t[2], tuple_group)
    non_type_by_pred = Index(state_non_type, lambda t: t[1], tuple_group)
    domain_by_subj = Index(tbox_domain, lambda t: t[0], tuple_group)
    range_by_subj = Index(tbox_range, lambda t: t[0], tuple_group)

    smj = lambda a, b, proj: DLDSortMergeJoin(a, b, proj, tuple_group, lattice)
    sco_tc = smj(sco_by_dst, sco_by_src, lambda _k, l, r: (l[0], SCO, r[2]))
    spo_tc = smj(spo_by_dst, spo_by_src, lambda _k, l, r: (l[0], SPO, r[2]))
    prop = smj(spo_by_src, non_type_by_pred, lambda _k, s, a: (a[0], s[2], a[2]))
    class_sco = smj(sco_by_src, type_by_obj, lambda _k, s, ty: (ty[0], TYPE, s[2]))
    domain = smj(domain_by_subj, non_type_by_pred, lambda _k, d, a: (a[0], TYPE, d[2]))
    range_ = smj(range_by_subj, non_type_by_pred, lambda _k, r, a: (a[2], TYPE, r[2]))

    fresh = sco_tc + spo_tc + prop + class_sco + domain + range_
    seeded = fresh + tbox_2d + abox_2d
    body_out = DLDDistinct(seeded, tuple_group, lattice)
    observable = TimeAxisElimination(body_out)

    evaluator = Evaluator(observable, parallelism=parallelism)
    setattr(observable, "_evaluator", evaluator)
    return RDFSCircuit(
        observable=observable,
        body_out=body_out,
        evaluator=evaluator,
        state=state,
        abox=abox,
        tbox=tbox,
        tuple_group=tuple_group,
        lattice=lattice,
    )


__all__ = ["IncrementalRDFSMaterializationWithIndexing"]
