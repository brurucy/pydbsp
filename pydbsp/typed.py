"""Typed 2-D DBSP program interface.

This layer keeps DBSP operator names visible while hiding the repetitive
``Time1D``/``Time2D``/``Input``/``Evaluator`` wiring. It is intended for
non-iterative streaming relational queries: every source delta is pushed
on the outer axis and lifted into a real 2-D DBSP stream internally.

Public operator functions build a logical expression tree. Registering a
view normalizes that tree, compiles it to physical DBSP streams, and emits
at most one physical distinct at the view root. ``DeltaLiftedDistinct`` is
therefore a final set-valued result constraint in this layer, not a local
materialization barrier.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, cast

from pydbsp.core import AbelianGroupOperation, Antichain, Time1D, Time2D
from pydbsp.evaluator import Evaluator
from pydbsp.indexed_zset.operators.bilinear import DLDSortMergeJoin
from pydbsp.indexed_zset.operators.linear import Index
from pydbsp.stream import Input, Lift1, Stream
from pydbsp.stream.functions.linear import TimeAxisIntroduction
from pydbsp.stream.operators.linear import Differentiate, Integrate
from pydbsp.stream.zset.operators.bilinear import DLDJoin
from pydbsp.stream.zset.operators.binary import DLDDistinct
from pydbsp.stream.zset.operators.linear import Project, Select
from pydbsp.zset import ZSet, ZSetAddition


T = TypeVar("T")
L = TypeVar("L")
R = TypeVar("R")
K = TypeVar("K")
O = TypeVar("O")
V = TypeVar("V")

Time = tuple[int, int]


class _Expr(Generic[T]):
    pass


@dataclass(frozen=True)
class _SourceExpr(_Expr[T]):
    stream: Stream[ZSet[T], Time]
    group: ZSetAddition[T]


@dataclass(frozen=True)
class _SelectExpr(_Expr[T]):
    input: _Expr[T]
    predicate: Callable[[T], bool]


@dataclass(frozen=True)
class _ProjectExpr(_Expr[O]):
    input: _Expr[T]
    projection: Callable[[T], O]
    group: ZSetAddition[O]


@dataclass(frozen=True)
class _AddExpr(_Expr[T]):
    left: _Expr[T]
    right: _Expr[T]


@dataclass(frozen=True)
class _SubtractExpr(_Expr[T]):
    left: _Expr[T]
    right: _Expr[T]


@dataclass(frozen=True)
class _NestedJoinExpr(_Expr[O]):
    left: _Expr[L]
    right: _Expr[R]
    predicate: Callable[[L, R], bool]
    projection: Callable[[L, R], O]
    group: ZSetAddition[O]


@dataclass(frozen=True)
class _SortMergeJoinExpr(_Expr[O]):
    left: _Expr[L]
    right: _Expr[R]
    left_key: Callable[[L], K]
    right_key: Callable[[R], K]
    projection: Callable[[L, R], O]
    group: ZSetAddition[O]


@dataclass(frozen=True)
class _AggregateExpr(_Expr[O]):
    input: _Expr[T]
    aggregate: Callable[[ZSet[T]], ZSet[O]]
    group: ZSetAddition[O]


@dataclass(frozen=True)
class _DistinctExpr(_Expr[T]):
    input: _Expr[T]


@dataclass(frozen=True)
class _Normalized(Generic[T]):
    expr: _Expr[T]
    distinct: bool


@dataclass(frozen=True)
class _Compiled(Generic[T]):
    stream: Stream[ZSet[T], Time]
    group: ZSetAddition[T]


class _TupleGroup(AbelianGroupOperation[tuple[Any, ...]]):
    def __init__(self, groups: tuple[AbelianGroupOperation[Any], ...]) -> None:
        self._groups = groups

    def add(self, a: tuple[Any, ...], b: tuple[Any, ...]) -> tuple[Any, ...]:
        return tuple(g.add(x, y) for g, x, y in zip(self._groups, a, b))

    def neg(self, a: tuple[Any, ...]) -> tuple[Any, ...]:
        return tuple(g.neg(x) for g, x in zip(self._groups, a))

    def identity(self) -> tuple[Any, ...]:
        return tuple(g.identity() for g in self._groups)


class _MultiOutputRoot(Stream[tuple[Any, ...], Time]):
    _stream_attrs = ("_streams",)

    def __init__(self, streams: tuple[Stream[Any, Time], ...]) -> None:
        self._streams = streams
        self._group = _TupleGroup(tuple(cast(AbelianGroupOperation[Any], s.group) for s in streams))

    @property
    def group(self) -> AbelianGroupOperation[tuple[Any, ...]]:
        return self._group

    @property
    def time_lattice(self):
        return Time2D

    @property
    def settled_frontier(self) -> Antichain[Time]:
        if not self._streams:
            return Antichain.universal(Time2D)
        frontier = self._streams[0].settled_frontier
        for stream in self._streams[1:]:
            frontier = frontier.meet(stream.settled_frontier)
        return frontier

    def _compute(self, t: Time) -> tuple[Any, ...]:
        return tuple(stream.at(t) for stream in self._streams)

    def deps(self, t: Time):
        return [(stream, t) for stream in self._streams]

    def compute_from(self, t: Time, slots):
        return tuple(slots[(id(stream), t)] for stream in self._streams)


@dataclass(eq=False)
class Stream2D(Generic[T]):
    _expr: _Expr[T]
    _group: ZSetAddition[T]


@dataclass(eq=False)
class Source(Stream2D[T]):
    name: str
    _input: Input[ZSet[T], tuple[int]]


@dataclass(eq=False)
class View(Generic[T]):
    name: str
    _program: "Program2D"
    _query: Stream2D[T]
    _stream: Stream[ZSet[T], Time]
    _group: ZSetAddition[T]
    _materialized: ZSet[T]
    _last_delta: ZSet[T]

    def delta(self) -> ZSet[T]:
        return _copy_zset(self._last_delta)

    def materialized(self) -> ZSet[T]:
        return _copy_zset(self._materialized)


def _copy_zset(value: ZSet[T]) -> ZSet[T]:
    return ZSet(dict(value.inner))


def _rows_to_zset(rows: ZSet[T] | Iterable[T] | None) -> ZSet[T]:
    if rows is None:
        return ZSet({})
    if isinstance(rows, ZSet):
        return rows
    values: dict[T, int] = {}
    for row in rows:
        values[row] = values.get(row, 0) + 1
    return ZSet({k: v for k, v in values.items() if v != 0})


def _negate_zset(value: ZSet[T]) -> ZSet[T]:
    return ZSet({k: -v for k, v in value.inner.items()})


def _normalize_expr(expr: _Expr[T]) -> _Normalized[T]:
    if isinstance(expr, _DistinctExpr):
        normalized = _normalize_expr(expr.input)
        return _Normalized(normalized.expr, True)
    if isinstance(expr, _SourceExpr):
        return _Normalized(expr, False)
    if isinstance(expr, _SelectExpr):
        normalized = _normalize_expr(expr.input)
        return _Normalized(_SelectExpr(normalized.expr, expr.predicate), normalized.distinct)
    if isinstance(expr, _ProjectExpr):
        normalized = _normalize_expr(expr.input)
        return _Normalized(_ProjectExpr(normalized.expr, expr.projection, expr.group), normalized.distinct)
    if isinstance(expr, _AggregateExpr):
        normalized = _normalize_expr(expr.input)
        return _Normalized(_AggregateExpr(normalized.expr, expr.aggregate, expr.group), normalized.distinct)
    if isinstance(expr, _AddExpr):
        left = _normalize_expr(expr.left)
        right = _normalize_expr(expr.right)
        return _Normalized(_AddExpr(left.expr, right.expr), left.distinct or right.distinct)
    if isinstance(expr, _SubtractExpr):
        left = _normalize_expr(expr.left)
        right = _normalize_expr(expr.right)
        return _Normalized(_SubtractExpr(left.expr, right.expr), left.distinct or right.distinct)
    if isinstance(expr, _NestedJoinExpr):
        left = _normalize_expr(expr.left)
        right = _normalize_expr(expr.right)
        return _Normalized(
            _NestedJoinExpr(left.expr, right.expr, expr.predicate, expr.projection, expr.group),
            left.distinct or right.distinct,
        )
    if isinstance(expr, _SortMergeJoinExpr):
        left = _normalize_expr(expr.left)
        right = _normalize_expr(expr.right)
        return _Normalized(
            _SortMergeJoinExpr(
                left.expr,
                right.expr,
                expr.left_key,
                expr.right_key,
                expr.projection,
                expr.group,
            ),
            left.distinct or right.distinct,
        )
    raise TypeError(f"unsupported typed DBSP expression: {type(expr).__name__}")


def _compile_expr(expr: _Expr[T], memo: dict[int, _Compiled[Any]] | None = None) -> _Compiled[T]:
    if memo is None:
        memo = {}
    cached = memo.get(id(expr))
    if cached is not None:
        return cast(_Compiled[T], cached)

    if isinstance(expr, _SourceExpr):
        compiled: _Compiled[Any] = _Compiled(expr.stream, expr.group)
    elif isinstance(expr, _SelectExpr):
        input_ = _compile_expr(expr.input, memo)
        compiled = _Compiled(Select(input_.stream, expr.predicate), input_.group)
    elif isinstance(expr, _ProjectExpr):
        input_ = _compile_expr(expr.input, memo)
        compiled = _Compiled(Project(input_.stream, expr.projection, expr.group), expr.group)
    elif isinstance(expr, _AddExpr):
        left = _compile_expr(expr.left, memo)
        right = _compile_expr(expr.right, memo)
        compiled = _Compiled(left.stream + right.stream, left.group)
    elif isinstance(expr, _SubtractExpr):
        left = _compile_expr(expr.left, memo)
        right = _compile_expr(expr.right, memo)
        compiled = _Compiled(left.stream - right.stream, left.group)
    elif isinstance(expr, _NestedJoinExpr):
        left = _compile_expr(expr.left, memo)
        right = _compile_expr(expr.right, memo)
        compiled = _Compiled(
            DLDJoin(left.stream, right.stream, expr.predicate, expr.projection, expr.group, Time2D),
            expr.group,
        )
    elif isinstance(expr, _SortMergeJoinExpr):
        left = _compile_expr(expr.left, memo)
        right = _compile_expr(expr.right, memo)
        left_indexed = Index(left.stream, expr.left_key, left.group)
        right_indexed = Index(right.stream, expr.right_key, right.group)
        compiled = _Compiled(
            DLDSortMergeJoin(
                left_indexed,
                right_indexed,
                lambda _key, l, r: expr.projection(l, r),
                expr.group,
                Time2D,
            ),
            expr.group,
        )
    elif isinstance(expr, _AggregateExpr):
        input_ = _compile_expr(expr.input, memo)
        cumulative = Integrate(input_.stream, Time2D, axis=0)
        lifted = Lift1(cumulative, expr.aggregate, expr.group)
        compiled = _Compiled(Differentiate(lifted, Time2D, axis=0), expr.group)
    elif isinstance(expr, _DistinctExpr):
        normalized = _normalize_expr(expr)
        compiled = _compile_expr(normalized.expr, memo)
        if normalized.distinct:
            compiled = _Compiled(DLDDistinct(compiled.stream, compiled.group, Time2D), compiled.group)
    else:
        raise TypeError(f"unsupported typed DBSP expression: {type(expr).__name__}")

    memo[id(expr)] = compiled
    return cast(_Compiled[T], compiled)


def _compile_view_expr(expr: _Expr[T]) -> _Compiled[T]:
    normalized = _normalize_expr(expr)
    compiled = _compile_expr(normalized.expr)
    if normalized.distinct:
        return _Compiled(DLDDistinct(compiled.stream, compiled.group, Time2D), compiled.group)
    return compiled


class Program2D:
    """A fixed 2-D DBSP program with one shared evaluator.

    Sources and views may be registered until the first ``step`` or
    explicit ``freeze``. After that the graph is fixed; each ``step``
    pushes a new outer delta and fills every registered view at ``(t, 0)``
    using the same evaluator and slot cache.
    """

    def __init__(
        self,
        *,
        gc: bool = True,
        parallelism: int = 1,
        parallel_layer_min_width: int = 4,
    ) -> None:
        self._gc = gc
        self._parallelism = parallelism
        self._parallel_layer_min_width = parallel_layer_min_width
        self._sources: list[Source[Any]] = []
        self._views: list[View[Any]] = []
        self._pending: dict[Source[Any], ZSet[Any]] = {}
        self._tick = 0
        self._frozen = False
        self._root: _MultiOutputRoot | None = None
        self._evaluator: Evaluator | None = None

    @property
    def tick(self) -> int:
        return self._tick

    @property
    def frozen(self) -> bool:
        return self._frozen

    @property
    def evaluator(self) -> Evaluator:
        if self._evaluator is None:
            raise RuntimeError("program has not been frozen yet")
        return self._evaluator

    def source(self, name: str) -> Source[T]:
        if self._frozen:
            raise RuntimeError("cannot add sources after Program2D is frozen")
        group: ZSetAddition[T] = ZSetAddition()
        source_input: Input[ZSet[T], tuple[int]] = Input(group, Time1D)
        lifted = TimeAxisIntroduction(source_input, group, Time2D, axis=1)
        source = Source(_expr=_SourceExpr(lifted, group), _group=group, name=name, _input=source_input)
        self._sources.append(cast(Source[Any], source))
        return source

    def view(self, name: str, query: Stream2D[T]) -> View[T]:
        if self._frozen:
            raise RuntimeError("cannot add views after Program2D is frozen")
        compiled = _compile_view_expr(query._expr)
        empty = compiled.group.identity()
        view = View(
            name=name,
            _program=self,
            _query=query,
            _stream=compiled.stream,
            _group=compiled.group,
            _materialized=empty,
            _last_delta=empty,
        )
        self._views.append(cast(View[Any], view))
        return view

    def insert(self, source: Source[T], rows: ZSet[T] | Iterable[T]) -> None:
        self._add_pending(source, _rows_to_zset(rows))

    def remove(self, source: Source[T], rows: ZSet[T] | Iterable[T]) -> None:
        self._add_pending(source, _negate_zset(_rows_to_zset(rows)))

    def freeze(self) -> None:
        if self._frozen:
            return
        if not self._views:
            raise RuntimeError("Program2D requires at least one registered view")
        self._root = _MultiOutputRoot(tuple(view._stream for view in self._views))
        self._evaluator = Evaluator(
            self._root,
            parallelism=self._parallelism,
            parallel_layer_min_width=self._parallel_layer_min_width,
            gc=self._gc,
        )
        self._frozen = True

    def step(
        self,
        deltas: Mapping[Source[Any], ZSet[Any] | Iterable[Any]] | None = None,
    ) -> dict[View[Any], ZSet[Any]]:
        self.freeze()
        outer_tick = self._tick
        source_deltas: dict[Source[Any], ZSet[Any]] = {}
        if deltas is not None:
            for source, rows in deltas.items():
                self._assert_source(source)
                source_deltas[source] = _rows_to_zset(rows)
        for source, pending_delta in self._pending.items():
            existing = source_deltas.get(source, source._group.identity())
            source_deltas[source] = source._group.add(existing, pending_delta)
        self._pending.clear()

        for source in self._sources:
            source._input.push((outer_tick,), source_deltas.get(source, source._group.identity()))

        targets = [(view._stream, (outer_tick, 0)) for view in self._views]
        assert self._evaluator is not None
        self._evaluator.fill_many(targets)

        out: dict[View[Any], ZSet[Any]] = {}
        for view in self._views:
            delta = cast(ZSet[Any], self._evaluator.slots[(id(view._stream), (outer_tick, 0))])
            view._last_delta = _copy_zset(delta)
            view._materialized = view._group.add(view._materialized, delta)
            out[view] = view.delta()
        self._tick += 1
        return out

    def gc(self) -> int:
        return self.evaluator.gc()

    def _add_pending(self, source: Source[T], delta: ZSet[T]) -> None:
        self._assert_source(source)
        existing = cast(ZSet[T], self._pending.get(source, source._group.identity()))
        self._pending[cast(Source[Any], source)] = source._group.add(existing, delta)

    def _assert_source(self, source: Source[Any]) -> None:
        if source not in self._sources:
            raise ValueError(f"source {source.name!r} does not belong to this Program2D")


def LiftedLiftedSelect(stream: Stream2D[T], predicate: Callable[[T], bool]) -> Stream2D[T]:
    return Stream2D(_SelectExpr(stream._expr, predicate), stream._group)


def LiftedLiftedProject(stream: Stream2D[T], projection: Callable[[T], O]) -> Stream2D[O]:
    out_group: ZSetAddition[O] = ZSetAddition()
    return Stream2D(_ProjectExpr(stream._expr, projection, out_group), out_group)


def DeltaLiftedSelectDistinct(stream: Stream2D[T], predicate: Callable[[T], bool]) -> Stream2D[T]:
    return DeltaLiftedDistinct(LiftedLiftedSelect(stream, predicate))


def DeltaLiftedProjectDistinct(stream: Stream2D[T], projection: Callable[[T], O]) -> Stream2D[O]:
    return DeltaLiftedDistinct(LiftedLiftedProject(stream, projection))


def LiftedLiftedAdd(left: Stream2D[T], right: Stream2D[T]) -> Stream2D[T]:
    return Stream2D(_AddExpr(left._expr, right._expr), left._group)


def LiftedLiftedSubtract(left: Stream2D[T], right: Stream2D[T]) -> Stream2D[T]:
    return Stream2D(_SubtractExpr(left._expr, right._expr), left._group)


def DeltaLiftedUnion(left: Stream2D[T], right: Stream2D[T]) -> Stream2D[T]:
    return DeltaLiftedDistinct(LiftedLiftedAdd(left, right))


def DeltaLiftedExcept(left: Stream2D[T], right: Stream2D[T]) -> Stream2D[T]:
    return DeltaLiftedDistinct(LiftedLiftedSubtract(left, right))


def DeltaLiftedDistinct(stream: Stream2D[T]) -> Stream2D[T]:
    return Stream2D(_DistinctExpr(stream._expr), stream._group)


def DeltaLiftedDeltaLiftedJoin(
    left: Stream2D[L],
    right: Stream2D[R],
    *,
    predicate: Callable[[L, R], bool],
    projection: Callable[[L, R], O],
) -> Stream2D[O]:
    out_group: ZSetAddition[O] = ZSetAddition()
    return Stream2D(
        _NestedJoinExpr(left._expr, right._expr, predicate, projection, out_group),
        out_group,
    )


def DeltaLiftedDeltaLiftedCartesianProduct(
    left: Stream2D[L],
    right: Stream2D[R],
    *,
    projection: Callable[[L, R], O],
) -> Stream2D[O]:
    return DeltaLiftedDeltaLiftedJoin(
        left,
        right,
        predicate=lambda _l, _r: True,
        projection=projection,
    )


def DeltaLiftedDeltaLiftedSortMergeJoin(
    left: Stream2D[L],
    right: Stream2D[R],
    *,
    left_key: Callable[[L], K],
    right_key: Callable[[R], K],
    projection: Callable[[L, R], O],
) -> Stream2D[O]:
    out_group: ZSetAddition[O] = ZSetAddition()
    return Stream2D(
        _SortMergeJoinExpr(
            left._expr,
            right._expr,
            left_key,
            right_key,
            projection,
            out_group,
        ),
        out_group,
    )


def DeltaLiftedDeltaLiftedEquiJoin(
    left: Stream2D[L],
    right: Stream2D[R],
    *,
    left_key: Callable[[L], K],
    right_key: Callable[[R], K],
    projection: Callable[[L, R], O],
) -> Stream2D[O]:
    return DeltaLiftedDeltaLiftedSortMergeJoin(
        left,
        right,
        left_key=left_key,
        right_key=right_key,
        projection=projection,
    )


def DeltaLiftedIntersect(
    left: Stream2D[T],
    right: Stream2D[T],
    *,
    key: Callable[[T], K] = lambda row: cast(Any, row),
) -> Stream2D[T]:
    return DeltaLiftedDistinct(
        Stream2D(
            _SortMergeJoinExpr(
                left._expr,
                right._expr,
                key,
                key,
                lambda l, _r: l,
                left._group,
            ),
            left._group,
        )
    )


def LiftedLiftedAggregate(
    stream: Stream2D[T],
    aggregate: Callable[[ZSet[T]], ZSet[O]],
) -> Stream2D[O]:
    """2-D version of DBSP's aggregate pattern.

    ``aggregate`` is evaluated over the cumulative input ZSet at each
    outer tick; the result is differentiated back into output deltas.
    The inner axis remains present and hidden by ``Program2D``.

    Aggregates operate on raw Z-set weights. A surrounding
    ``DeltaLiftedDistinct`` is normalized to the final view output, so
    aggregate functions that want set semantics should clamp/interpret
    weights explicitly.
    """
    out_group: ZSetAddition[O] = ZSetAddition()
    return Stream2D(_AggregateExpr(stream._expr, aggregate, out_group), out_group)


def LiftedLiftedGroupBySum(
    stream: Stream2D[T],
    *,
    key: Callable[[T], K],
    value: Callable[[T], int | float],
    output: Callable[[K, int | float], O],
) -> Stream2D[O]:
    def aggregate(zset: ZSet[T]) -> ZSet[O]:
        totals: dict[K, int | float] = {}
        for row, weight in zset.inner.items():
            totals[key(row)] = totals.get(key(row), 0) + value(row) * weight
        return ZSet({output(k, v): 1 for k, v in totals.items()})

    return LiftedLiftedAggregate(stream, aggregate)


def LiftedLiftedGroupByMax(
    stream: Stream2D[T],
    *,
    key: Callable[[T], K],
    value: Callable[[T], V],
    output: Callable[[K, V], O],
) -> Stream2D[O]:
    def aggregate(zset: ZSet[T]) -> ZSet[O]:
        max_by_key: dict[K, V] = {}
        for row, weight in zset.inner.items():
            if weight <= 0:
                continue
            k = key(row)
            v = value(row)
            cur = max_by_key.get(k)
            if cur is None or v > cur:  # type: ignore[operator]
                max_by_key[k] = v
        return ZSet({output(k, v): 1 for k, v in max_by_key.items()})

    return LiftedLiftedAggregate(stream, aggregate)


__all__ = [
    "DeltaLiftedDeltaLiftedCartesianProduct",
    "DeltaLiftedDeltaLiftedEquiJoin",
    "DeltaLiftedDeltaLiftedJoin",
    "DeltaLiftedDeltaLiftedSortMergeJoin",
    "DeltaLiftedExcept",
    "DeltaLiftedDistinct",
    "DeltaLiftedIntersect",
    "DeltaLiftedProjectDistinct",
    "DeltaLiftedSelectDistinct",
    "DeltaLiftedUnion",
    "LiftedLiftedAdd",
    "LiftedLiftedAggregate",
    "LiftedLiftedGroupByMax",
    "LiftedLiftedGroupBySum",
    "LiftedLiftedProject",
    "LiftedLiftedSelect",
    "LiftedLiftedSubtract",
    "Program2D",
    "Source",
    "Stream2D",
    "View",
]
