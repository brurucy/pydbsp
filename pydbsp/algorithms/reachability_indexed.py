"""Sort-merge-join variant of ``IncrementalReachability`` on a flat
product lattice.

Same topology as the plain variant; the difference is the join —
state is indexed by ``dst``, new edges by ``src``, and the 4-term
join runs sort-merge on the shared int key via
``DLDSortMergeJoin``.
"""


from dataclasses import dataclass

from pydbsp.core import DBSPTime, Time1D, Time2D
from pydbsp.evaluator import Evaluator
from pydbsp.indexed_zset.operators.bilinear import DLDSortMergeJoin
from pydbsp.indexed_zset.operators.linear import Index
from pydbsp.stream import Input, Stream
from pydbsp.stream.functions.linear import TimeAxisElimination, TimeAxisIntroduction
from pydbsp.stream.operators.linear import Delay
from pydbsp.stream.zset.operators.binary import DLDDistinct
from pydbsp.zset import ZSet, ZSetAddition


Edge = tuple[int, int]
Time = tuple[int, int]


@dataclass
class ReachabilityCircuit:
    observable: Stream[ZSet[Edge], tuple[int]]
    body_out: Stream[ZSet[Edge], Time]
    state: Input[ZSet[Edge], Time]
    edges: Input[ZSet[Edge], tuple[int]]
    edge_group: ZSetAddition[Edge]
    lattice: DBSPTime[Time]
    evaluator: Evaluator

    def body_at(self, t: Time) -> ZSet[Edge]:
        """Returns the per-iteration diff at the 2-D timestamp ``t``."""
        return self.evaluator.at_op(self.body_out, t)

    def observable_at(self, t: tuple[int]) -> ZSet[Edge]:
        """Returns the reachability output delta at outer timestamp ``t``."""
        return self.evaluator.at(t)


def saturate_reach(
    circuit: ReachabilityCircuit,
    outer_tick: int = 0,
    *,
    max_inner: int = 1 << 16,
) -> int:
    """Drives the inner fixpoint at ``outer_tick`` to convergence. Returns
    the final inner tick. Prior outer ticks' inner cells are padded with
    identity so inner frontiers stay aligned across outers.
    """
    state = circuit.state
    prior_max = -1
    for (o, k) in state._values.keys():
        if o < outer_tick and k > prior_max:
            prior_max = k

    k_final = -1
    for k in range(max_inner):
        diff = circuit.body_at((outer_tick, k))
        is_zero = diff.inner == {}
        state.push((outer_tick, k), diff)
        if is_zero and k > prior_max:
            k_final = k
            break
    else:
        raise RuntimeError(f"iteration did not converge in {max_inner} inner ticks")

    for o_prior in range(outer_tick):
        for k_pad in range(prior_max + 1, k_final + 1):
            if (o_prior, k_pad) not in state._values:
                state.push((o_prior, k_pad), state.group.identity())
    return k_final


def IncrementalReachabilityWithIndexing(parallelism: int = 1) -> ReachabilityCircuit:
    """Builds an incremental graph-reachability circuit with sort-merge-
    join indexing. Push edges at outer ticks, call ``saturate_reach``,
    read the closure via ``observable_at``. Faster than
    ``IncrementalReachability`` on larger graphs.

    ``parallelism`` >1 dispatches each P-antichain layer through a
    thread pool; real speedup requires free-threaded Python (3.14t,
    ``PYTHON_GIL=0``).
    """
    edge_group: ZSetAddition[Edge] = ZSetAddition()
    lattice = Time2D

    # Edges are a 1-D stream (vary with the outer tick, constant
    # across inner iterations at each outer). Introduce into the 2-D
    # body lattice with a universal frontier along the inner axis so
    # the fixpoint loop doesn't have to pad the inner axis by hand.
    edges = Input(edge_group, Time1D)
    edges_2d = TimeAxisIntroduction(edges, edge_group, lattice, axis=1)
    state = Input(edge_group, lattice)

    hop = DLDSortMergeJoin(
        Index(Delay(state, lattice, axis=1), lambda e: e[1], edge_group),
        Index(edges_2d, lambda e: e[0], edge_group),
        lambda k, l, r: (l[0], r[1]),
        edge_group,
        lattice,
    )
    fresh = hop + edges_2d
    body_out = DLDDistinct(fresh, edge_group, lattice)
    observable = TimeAxisElimination(body_out)

    evaluator = Evaluator(observable, parallelism=parallelism)
    setattr(observable, "_evaluator", evaluator)
    return ReachabilityCircuit(
        observable=observable,
        body_out=body_out,
        evaluator=evaluator,
        state=state,
        edges=edges,
        edge_group=edge_group,
        lattice=lattice,
    )
