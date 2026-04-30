"""Execution driver — detaches *when* a circuit's values become
observable from *what* the circuit computes.

The circuit is a pure graph of operators (frontier rule + value rule
per operator), described by a ``Stream`` (its output). ``History``
owns a cursor antichain on the circuit's ``DBSPTime`` lattice and
exposes ``try_step`` to extend that cursor one successor along a
chosen axis. The move is accepted iff the circuit's output frontier
already admits every proposed point; otherwise the cursor is left
untouched and ``try_step`` reports ``False``.

Reads go through ``at``, which checks the cursor before delegating
to the schedule-driven ``Evaluator`` — the causality invariant
(no reads outside the settled region) becomes an invariant of the
driver, not a per-stream assertion.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydbsp.core import Antichain, DBSPTime
from pydbsp.stream import Stream

if TYPE_CHECKING:
    from pydbsp.evaluator import Evaluator


class History[V, T: tuple[int, ...]]:
    """Observer over a circuit.

    Construction takes the circuit's output stream and records its
    time lattice (must be a ``DBSPTime[T]``). The cursor starts
    empty — no reads are legal until the first ``try_step`` commits
    to the lattice's ``bottom`` point.

    ``try_step(axis)`` proposes the antichain-level successor on the
    given axis (per ``DBSPTime.advance_antichain``). The proposal is
    accepted iff it is contained in the circuit output's
    ``settled_frontier``. On accept, the cursor is updated; on refuse,
    nothing happens and ``False`` is returned.

    ``at(t)`` is a cursor-gated read: legal only when ``t`` is
    covered by the cursor. Internally delegates to an ``Evaluator``
    — iterative schedule-driven fill with direct slot-table lookups.
    """

    def __init__(self, output: Stream[V, T], evaluator: "Evaluator | None" = None) -> None:
        lattice = output.time_lattice
        if not isinstance(lattice, DBSPTime):
            raise TypeError(
                f"History requires the circuit output's time lattice to be a DBSPTime; got {type(lattice).__name__}"
            )
        self._output: Stream[V, T] = output
        self._lattice: DBSPTime[T] = lattice
        self._frontier: Antichain[T] = Antichain(self._lattice)
        self._eval = evaluator or getattr(output, "_evaluator", None)

    @property
    def lattice(self) -> DBSPTime[T]:
        return self._lattice

    @property
    def frontier(self) -> Antichain[T]:
        return self._frontier

    def try_step(self, axis: int = 0) -> bool:
        if axis < 0 or axis >= self._lattice.nestedness:
            raise IndexError(f"axis {axis} out of range for lattice of nestedness {self._lattice.nestedness}")
        proposed = self._lattice.advance_antichain(self._frontier, axis)
        if not proposed.leq(self._output.settled_frontier):
            return False
        self._frontier = proposed
        return True

    def at(self, t: T) -> V:
        if not self._frontier.covers(t):
            raise IndexError(f"{t} outside cursor")
        if self._eval is None:
            from pydbsp.evaluator import Evaluator
            self._eval = Evaluator(self._output)
        return self._eval.at(t)


@dataclass
class Schedule:
    """Static compile-time view of a circuit's DAG.

    Produced by ``compile_schedule``. ``ops`` is topologically ordered
    (leaves first, root last). ``deps[i]`` holds upstream op indices
    for op ``i``. ``layer_of[i]`` is the op's depth from leaves;
    ``layers[k]`` gives the op indices at depth ``k`` — all mutually
    independent within a layer, the natural parallel unit.

    ``critical_path`` is one longest leaf-to-root chain; its length
    is the serial lower bound on wall time. Layer widths bound the
    achievable parallelism.
    """

    ops: list[Stream]
    deps: dict[int, list[int]]
    layer_of: dict[int, int]
    layers: list[list[int]]
    root: int
    critical_path: list[int]

    @property
    def depth(self) -> int:
        return len(self.layers)

    @property
    def widths(self) -> list[int]:
        return [len(layer) for layer in self.layers]

    @property
    def max_width(self) -> int:
        return max(self.widths, default=0)


def compile_schedule[V, T: tuple[int, ...]](output: Stream[V, T]) -> Schedule:
    """Walk the DAG rooted at ``output`` via ``_stream_attrs``, collect
    operators in topological order (post-order DFS), compute per-op
    layer = 1 + max(dep layers), and extract one longest critical
    path by greedy backward walk from the root.
    """
    ops: list[Stream] = []
    op_index: dict[int, int] = {}
    deps: dict[int, list[int]] = {}

    def visit(s: Stream) -> int:
        sid = id(s)
        if sid in op_index:
            return op_index[sid]
        dep_indices: list[int] = []
        for attr in getattr(type(s), "_stream_attrs", ()):
            child = getattr(s, attr)
            if isinstance(child, (tuple, list)):
                for c in child:
                    dep_indices.append(visit(c))
            else:
                dep_indices.append(visit(child))
        idx = len(ops)
        op_index[sid] = idx
        ops.append(s)
        deps[idx] = dep_indices
        return idx

    root = visit(output)

    layer_of: dict[int, int] = {}
    for idx in range(len(ops)):
        dep_layers = [layer_of[d] for d in deps[idx]]
        layer_of[idx] = 1 + max(dep_layers) if dep_layers else 0

    max_layer = max(layer_of.values(), default=0)
    layers: list[list[int]] = [[] for _ in range(max_layer + 1)]
    for idx, layer in layer_of.items():
        layers[layer].append(idx)

    critical: list[int] = [root]
    current = root
    while deps[current]:
        current = max(deps[current], key=lambda d: layer_of[d])
        critical.append(current)
    critical.reverse()

    return Schedule(ops=ops, deps=deps, layer_of=layer_of, layers=layers, root=root, critical_path=critical)
