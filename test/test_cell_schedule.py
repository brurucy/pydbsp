"""Cell-level P-antichain schedule: correctness + structural properties.

P-antichain is the only dispatch strategy the ``Evaluator`` ships.
These tests check that running end-to-end workloads produces correct
answers, and that the compiled schedule has the structural properties
we expect (topologically sorted layers, deterministic, tighter than
the op-level ``Schedule`` would be as a wavefront dispatcher).
"""

from __future__ import annotations

from pathlib import Path

from pydbsp.algorithms import Variable
from pydbsp.algorithms.datalog import saturate
from pydbsp.algorithms.datalog_indexed import IncrementalDatalogWithIndexing
from pydbsp.algorithms.reachability_indexed import (
    IncrementalReachabilityWithIndexing,
    saturate_reach,
)
from pydbsp.evaluator import CellSchedule, Evaluator, compile_cell_schedule
from pydbsp.stream import Input
from pydbsp.core import Time1D
from pydbsp.stream.operators.linear import Integrate
from pydbsp.zset import ZSet, ZSetAddition


def _graph1000_edges() -> ZSet:
    edges: dict = {}
    for line in Path("notebooks/data/graph1000.txt").read_text().splitlines():
        p = line.split()
        if len(p) >= 2:
            edges[(int(p[0]), int(p[1]))] = 1
    return ZSet(edges)


# ---- correctness ------------------------------------------------------------


def test_p_antichain_running_sum() -> None:
    """1-D Integrate: cumulative values across three input ticks."""
    g = ZSetAddition[int]()
    inp = Input(g, Time1D)
    cum = Integrate(inp, Time1D, axis=0)
    ev = Evaluator(cum)
    inp.push((0,), ZSet({1: 1}))
    inp.push((1,), ZSet({2: 1}))
    inp.push((2,), ZSet({1: 1}))
    assert ev.at((0,)).inner == {1: 1}
    assert ev.at((1,)).inner == {1: 1, 2: 1}
    assert ev.at((2,)).inner == {1: 2, 2: 1}


def test_p_antichain_reach_graph1000() -> None:
    """Reachability saturation converges to the known closure size."""
    c = IncrementalReachabilityWithIndexing()
    c.edges.push((0,), _graph1000_edges())
    saturate_reach(c)
    assert len(c.observable_at((0,)).inner) == 11532


def test_p_antichain_datalog_tc_graph1000() -> None:
    """Indexed Datalog TC agrees with the native reach closure."""
    program = ZSet(
        {
            (
                ("T", (Variable("X"), Variable("Y"))),
                ("E", (Variable("X"), Variable("Y"))),
            ): 1,
            (
                ("T", (Variable("X"), Variable("Z"))),
                ("E", (Variable("X"), Variable("Y"))),
                ("T", (Variable("Y"), Variable("Z"))),
            ): 1,
        }
    )
    edges = _graph1000_edges()
    edb = ZSet({("E", e): 1 for e in edges.inner})
    c = IncrementalDatalogWithIndexing()
    c.edb.push((0,), edb)
    c.program.push((0,), program)
    saturate(c)
    t_facts = {k[1] for k in c.observable_at((0,)).inner if k[0] == "T"}
    # Full TC must match reach's 11532 pairs; EDB is a subset.
    assert len(t_facts) == 11532
    assert set(edges.inner).issubset(t_facts)


# ---- structural properties --------------------------------------------------


def test_cell_schedule_layers_are_topologically_sorted() -> None:
    """Every dep of a layer-L cell lies in layers 0..L-1."""
    c = IncrementalReachabilityWithIndexing()
    c.edges.push((0,), ZSet({(1, 2): 1, (2, 3): 1, (3, 4): 1}))
    saturate_reach(c)

    sched = compile_cell_schedule(
        c.evaluator, [(c.observable, (0,))], skip_computed=False
    )
    layer_of = {cell: L for L, layer in enumerate(sched.layers) for cell in layer}
    op_idx = c.evaluator._op_idx
    ops_list = c.evaluator._schedule.ops
    for L, layer in enumerate(sched.layers):
        for op_i, t in layer:
            op = ops_list[op_i]
            for dop, dt in op.deps(t):
                dep_key = (op_idx[id(dop)], dt)
                assert dep_key in layer_of, (
                    f"dep {dep_key} of {(op_i, t)} not in schedule"
                )
                assert layer_of[dep_key] < L, (
                    f"dep {dep_key} at layer {layer_of[dep_key]} "
                    f"not before {(op_i, t)} at layer {L}"
                )


def test_cell_schedule_is_tighter_than_op_level_wavefront() -> None:
    """Cell schedule packs the same work into many fewer dispatch steps
    than an op-level ``(Schedule layer, sum(t))`` wavefront would.
    """
    program = ZSet(
        {
            (
                ("T", (Variable("X"), Variable("Y"))),
                ("E", (Variable("X"), Variable("Y"))),
            ): 1,
            (
                ("T", (Variable("X"), Variable("Z"))),
                ("E", (Variable("X"), Variable("Y"))),
                ("T", (Variable("Y"), Variable("Z"))),
            ): 1,
        }
    )
    edb = ZSet({("E", (i, i + 1)): 1 for i in range(4)})

    c = IncrementalDatalogWithIndexing()
    c.edb.push((0,), edb)
    c.program.push((0,), program)
    saturate(c)
    _ = c.observable_at((0,))

    ev = c.evaluator
    sched = compile_cell_schedule(
        ev, [(c.observable, (0,))], skip_computed=False
    )

    # Count what an op-level wavefront dispatcher would need — unique
    # (Schedule layer, sum(t)) pairs across the work set.
    op_idx = ev._op_idx
    op_to_layer = {
        op_i: L for L, layer in enumerate(ev._schedule.layers) for op_i in layer
    }
    work: set = set()
    stack = [(c.observable, (0,))]
    while stack:
        op, t = stack.pop()
        key = (op_idx[id(op)], t)
        if key in work:
            continue
        work.add(key)
        for dop, dt in op.deps(t):
            stack.append((dop, dt))
    wf_steps = len({(op_to_layer[op_i], sum(t)) for (op_i, t) in work})

    assert sched.n_cells == len(work), "same work"
    assert sched.n_layers < wf_steps, (
        f"cell schedule should be tighter: {sched.n_layers} vs {wf_steps}"
    )


def test_cell_schedule_is_deterministic() -> None:
    """Compiling twice from the same state yields the same layer structure."""
    c = IncrementalReachabilityWithIndexing()
    c.edges.push((0,), ZSet({(1, 2): 1, (2, 3): 1}))
    saturate_reach(c)

    s1 = compile_cell_schedule(c.evaluator, [(c.observable, (0,))])
    s2 = compile_cell_schedule(c.evaluator, [(c.observable, (0,))])
    assert [list(l) for l in s1.layers] == [list(l) for l in s2.layers]


def test_cell_schedule_skips_already_computed_cells() -> None:
    """Compiling after cells are in slots yields only the pending work."""
    c = IncrementalReachabilityWithIndexing()
    c.edges.push((0,), ZSet({(1, 2): 1, (2, 3): 1}))
    _ = c.body_at((0, 0))
    ev = c.evaluator
    full = compile_cell_schedule(
        ev, [(c.body_out, (0, 0))], skip_computed=False
    )
    partial = compile_cell_schedule(
        ev, [(c.body_out, (0, 0))], skip_computed=True
    )
    assert full.n_cells > 0
    assert partial.n_cells == 0


def test_cell_schedule_returns_cellschedule_type() -> None:
    g = ZSetAddition[int]()
    inp = Input(g, Time1D)
    cum = Integrate(inp, Time1D, axis=0)
    inp.push((0,), ZSet({1: 1}))
    ev = Evaluator(cum)
    sched = compile_cell_schedule(ev, [(cum, (0,))])
    assert isinstance(sched, CellSchedule)
    assert sched.n_cells > 0
    assert sched.n_layers > 0


def test_cell_schedule_handles_multiple_targets() -> None:
    """Multi-target compile collapses shared deps into a single schedule."""
    g = ZSetAddition[int]()
    inp = Input(g, Time1D)
    cum = Integrate(inp, Time1D, axis=0)
    inp.push((0,), ZSet({1: 1}))
    inp.push((1,), ZSet({2: 1}))
    inp.push((2,), ZSet({3: 1}))
    ev = Evaluator(cum)
    sched = compile_cell_schedule(ev, [(cum, (0,)), (cum, (1,)), (cum, (2,))])
    # cum at three ticks + their 3 Input deps = 6 distinct cells.
    assert sched.n_cells == 6
    assert sched.n_layers >= 2  # input cells in earlier layers, cum in later
