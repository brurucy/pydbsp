"""RDFS materialization on the flat 2-D product lattice.

RDFS is a constrained Datalog program. Its fixed six-rule set is
baked into a v2 body operator rather than going through the general
:class:`~pydbsp.relational_operators.IncrementalDatalogBody`
machinery. State is a single ``ZSet[RDFTuple]`` accumulating the
materialized ABox plus the TBox closures (SCO and SPO transitive
chains), distinguished by the predicate-position code:

* :data:`SCO`. SubClassOf
* :data:`SPO`. SubPropertyOf
* :data:`DOMAIN`. Rdfs:domain
* :data:`RANGE`. Rdfs:range
* :data:`TYPE`. Rdf:type

Six rules baked into the body:

* **SCO TC**:    ``(x, SCO, z) :- (x, SCO, y), (y, SCO, z)``
* **SPO TC**:    ``(x, SPO, z) :- (x, SPO, y), (y, SPO, z)``
* **Property**:  ``(x, b, y)   :- (a, SPO, b), (x, a, y)``
* **Class TC**:  ``(z, TYPE, y) :- (x, SCO, y), (z, TYPE, x)``
* **Domain**:    ``(y, TYPE, x) :- (a, DOMAIN, x), (y, a, z)``
* **Range**:     ``(z, TYPE, x) :- (a, RANGE, x), (y, a, z)``

All six joins are on a single integer field of the ``RDFTuple``. Natural sort-merge candidates. This module ports
``pydbsp.algorithms.rdfs_indexed.IncrementalRDFSMaterializationWithIndexing``
onto the v2 Circuit. The plain (non-indexed) variant is intentionally
not ported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydbsp.circuit import Circuit
from pydbsp.indexed_relational_operators import (
    IndexedDeltaLiftedDeltaLiftedJoin,
    LiftIndex,
)
from pydbsp.indexed_zset import IndexedZSetAddition
from pydbsp.operator import Lift2, LiftDelay, LiftStreamIntroduction, Operator
from pydbsp.progress import NodeId, Time
from pydbsp.relational_operators import (
    DeltaLiftedDeltaLiftedDistinct,
    LiftSelect,
)
from pydbsp.zset import ZSet, ZSetAddition


# ---- RDFS value-layer types and predicate codes ---------------------------

Subject = Any
Object = Any
Property = Any
RDFTuple = tuple[Subject, Property, Object]
RDFGraph = ZSet[RDFTuple]

SCO = 0  #: subClassOf predicate code
SPO = 1  #: subPropertyOf predicate code
DOMAIN = 2  #: rdfs:domain predicate code
RANGE = 3  #: rdfs:range predicate code
TYPE = 4  #: rdf:type predicate code


# ---- Body operator --------------------------------------------------------


@dataclass(frozen=True)
class IndexedIncrementalRDFSBody(Operator):
    """Body of an incremental RDFS materializer with sort-merge-join
    indexing on a 2-D flat product lattice (outer = batch progression,
    inner = fixpoint iteration). Mirrors
    :func:`pydbsp.algorithms.rdfs_indexed.IncrementalRDFSMaterializationWithIndexing`:
    six rules → six pre-indexed sort-merge joins, summed, then run
    through the doubly-incremental ``H``-distinct.

    ``inputs = (abox_1d, tbox_1d, state_2d)``.

    * ``abox_1d`` (``ZSet[RDFTuple]``, 1-D): ABox tuples pushed at
      outer ticks.
    * ``tbox_1d`` (``ZSet[RDFTuple]``, 1-D): TBox tuples (schema with
      ``SCO`` / ``SPO`` / ``DOMAIN`` / ``RANGE`` entries).
    * ``state_2d``: 2-D ``Input``. Caller pushes per-iteration body
      diffs back at explicit ``(outer, inner)`` coordinates via
      ``e.push(state_2d, value, t=(o, k))``.

    Returns a single ``NodeId`` whose value at each ``(outer, inner)``
    is the body diff. Drive the fixpoint via
    ``e.saturate_inner(body, outer_tick, is_empty=lambda d: not d.inner)``.

    The state-feedback cycle is **not** wired inside this operator.
    ``state_2d`` is expected to be an external Input whose pushes are
    driven by the caller's per-outer-tick fixpoint loop, exactly as
    with :class:`pydbsp.relational_operators.IncrementalReachabilityBody`.
    """

    tuple_group: ZSetAddition[RDFTuple]

    def connect(
        self,
        circuit: Circuit[Time],
        inputs: tuple[NodeId, ...],
    ) -> NodeId:
        if len(inputs) != 3:
            raise ValueError(f"IndexedIncrementalRDFSBody takes 3 inputs; got {len(inputs)}")
        abox_1d, tbox_1d, state_2d = inputs

        # ---- Lift 1-D ABox / TBox to 2-D ----------------------------------
        abox_2d = LiftStreamIntroduction[ZSet[RDFTuple]](group=self.tuple_group).connect(circuit, (abox_1d,))
        tbox_2d = LiftStreamIntroduction[ZSet[RDFTuple]](group=self.tuple_group).connect(circuit, (tbox_1d,))

        # ---- Predicate filters on the TBox --------------------------------
        tbox_domain = LiftSelect[RDFTuple](pred=lambda t: t[1] == DOMAIN).connect(circuit, (tbox_2d,))
        tbox_range = LiftSelect[RDFTuple](pred=lambda t: t[1] == RANGE).connect(circuit, (tbox_2d,))

        # ---- z⁻¹ⁱ on state, then split by predicate kind ------------------
        delayed_state = LiftDelay[ZSet[RDFTuple]](group=self.tuple_group).connect(circuit, (state_2d,))

        state_sco = LiftSelect[RDFTuple](pred=lambda t: t[1] == SCO).connect(circuit, (delayed_state,))
        state_spo = LiftSelect[RDFTuple](pred=lambda t: t[1] == SPO).connect(circuit, (delayed_state,))
        state_type = LiftSelect[RDFTuple](pred=lambda t: t[1] == TYPE).connect(circuit, (delayed_state,))
        state_non_type = LiftSelect[RDFTuple](pred=lambda t: t[1] not in (SCO, SPO, TYPE, DOMAIN, RANGE)).connect(
            circuit, (delayed_state,)
        )

        # ---- Indexed sides of the six joins -------------------------------
        # Each side is indexed on the column of RDFTuple that the join
        # matches on. The IndexedZSetAddition groups carry the same
        # key-extractor so the bilinear join can MGGroup-add per key.

        def _by(field_idx: int, source: NodeId) -> NodeId:
            return LiftIndex[RDFTuple, Any](indexer=lambda t, _i=field_idx: t[_i]).connect(circuit, (source,))

        def _idx_group(field_idx: int) -> IndexedZSetAddition[Any, RDFTuple]:
            return IndexedZSetAddition[Any, RDFTuple](self.tuple_group, lambda t, _i=field_idx: t[_i])

        sco_by_dst = _by(2, state_sco)
        sco_by_src = _by(0, state_sco)
        spo_by_dst = _by(2, state_spo)
        spo_by_src = _by(0, state_spo)
        type_by_obj = _by(2, state_type)
        non_type_by_pred = _by(1, state_non_type)
        domain_by_subj = _by(0, tbox_domain)
        range_by_subj = _by(0, tbox_range)

        idx_g0 = _idx_group(0)
        idx_g1 = _idx_group(1)
        idx_g2 = _idx_group(2)

        def _smj(
            left: NodeId,
            right: NodeId,
            group_a: IndexedZSetAddition[Any, RDFTuple],
            group_b: IndexedZSetAddition[Any, RDFTuple],
            proj,
        ) -> NodeId:
            return IndexedDeltaLiftedDeltaLiftedJoin[Any, RDFTuple, RDFTuple, RDFTuple](
                proj=proj,
                group_a=group_a,
                group_b=group_b,
                out_group=self.tuple_group,
            ).connect(circuit, (left, right))

        # ---- Six rules as indexed sort-merge joins -----------------------
        # SCO TC: (x, SCO, z) :- (x, SCO, y), (y, SCO, z)
        sco_tc = _smj(
            sco_by_dst,
            sco_by_src,
            idx_g2,
            idx_g0,
            lambda _k, l, r: (l[0], SCO, r[2]),
        )
        # SPO TC: (x, SPO, z) :- (x, SPO, y), (y, SPO, z)
        spo_tc = _smj(
            spo_by_dst,
            spo_by_src,
            idx_g2,
            idx_g0,
            lambda _k, l, r: (l[0], SPO, r[2]),
        )
        # Property: (x, b, y) :- (a, SPO, b), (x, a, y)
        prop = _smj(
            spo_by_src,
            non_type_by_pred,
            idx_g0,
            idx_g1,
            lambda _k, s, a: (a[0], s[2], a[2]),
        )
        # Class TC: (z, TYPE, y) :- (x, SCO, y), (z, TYPE, x)
        class_sco = _smj(
            sco_by_src,
            type_by_obj,
            idx_g0,
            idx_g2,
            lambda _k, s, ty: (ty[0], TYPE, s[2]),
        )
        # Domain: (y, TYPE, x) :- (a, DOMAIN, x), (y, a, z)
        domain = _smj(
            domain_by_subj,
            non_type_by_pred,
            idx_g0,
            idx_g1,
            lambda _k, d, a: (a[0], TYPE, d[2]),
        )
        # Range: (z, TYPE, x) :- (a, RANGE, x), (y, a, z)
        range_rule = _smj(
            range_by_subj,
            non_type_by_pred,
            idx_g0,
            idx_g1,
            lambda _k, r, a: (a[2], TYPE, r[2]),
        )

        # ---- Sum the six derivation streams + seed with TBox + ABox ------
        add = self.tuple_group.add

        def _sum(a: NodeId, b: NodeId) -> NodeId:
            return Lift2[ZSet[RDFTuple], ZSet[RDFTuple], ZSet[RDFTuple]](op=add).connect(circuit, (a, b))

        s = _sum(sco_tc, spo_tc)
        s = _sum(s, prop)
        s = _sum(s, class_sco)
        s = _sum(s, domain)
        s = _sum(s, range_rule)
        s = _sum(s, tbox_2d)
        seeded = _sum(s, abox_2d)

        # ---- Doubly-incremental distinct closes the body -----------------
        return DeltaLiftedDeltaLiftedDistinct[RDFTuple](inner_group=self.tuple_group).connect(circuit, (seeded,))
