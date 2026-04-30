from typing import NamedTuple

import pytest

from pydbsp.typed import (
    DeltaLiftedDeltaLiftedCartesianProduct,
    DeltaLiftedDeltaLiftedEquiJoin,
    DeltaLiftedDeltaLiftedJoin,
    DeltaLiftedDeltaLiftedSortMergeJoin,
    DeltaLiftedDistinct,
    DeltaLiftedExcept,
    DeltaLiftedIntersect,
    DeltaLiftedProjectDistinct,
    DeltaLiftedSelectDistinct,
    DeltaLiftedUnion,
    LiftedLiftedGroupByMax,
    LiftedLiftedGroupBySum,
    LiftedLiftedProject,
    LiftedLiftedSelect,
    LiftedLiftedSubtract,
    Program2D,
    Source,
    _AggregateExpr,
    _DistinctExpr,
    _normalize_expr,
)
from pydbsp.zset.functions.binary import H
from pydbsp.zset import ZSet


class Order(NamedTuple):
    id: int
    customer: int
    total: float


class Customer(NamedTuple):
    id: int
    country: str


class EEOrder(NamedTuple):
    order_id: int
    total: float


class KV(NamedTuple):
    key: str
    value: int


class Agg(NamedTuple):
    key: str
    value: int | float


def _distinct_kernel_count(program: Program2D) -> int:
    return sum(1 for op in program.evaluator.schedule.ops if getattr(op, "_f", None) is H)


def _contains_expr(expr: object, expr_type: type[object]) -> bool:
    if isinstance(expr, expr_type):
        return True
    for attr in ("input", "left", "right"):
        child = getattr(expr, attr, None)
        if child is not None and _contains_expr(child, expr_type):
            return True
    return False


def test_program2d_sort_merge_join_steps_incrementally_with_one_evaluator() -> None:
    p = Program2D()
    orders: Source[Order] = p.source("orders")
    customers: Source[Customer] = p.source("customers")

    joined = DeltaLiftedDeltaLiftedSortMergeJoin(
        orders,
        customers,
        left_key=lambda o: o.customer,
        right_key=lambda c: c.id,
        projection=lambda o, _c: EEOrder(o.id, o.total),
    )
    view = p.view("orders_by_customer", joined)

    p.step({orders: [Order(1, 7, 30.0)]})
    assert view.delta().inner == {}

    evaluator_id = id(p.evaluator)
    p.step({customers: [Customer(7, "EE")]})

    assert id(p.evaluator) == evaluator_id
    assert view.delta().inner == {EEOrder(1, 30.0): 1}
    assert view.materialized().inner == {EEOrder(1, 30.0): 1}


def test_program2d_nested_loop_join_and_lifted_lifted_ops() -> None:
    p = Program2D()
    orders: Source[Order] = p.source("orders")
    customers: Source[Customer] = p.source("customers")

    joined = DeltaLiftedDeltaLiftedJoin(
        orders,
        customers,
        predicate=lambda o, c: o.customer == c.id,
        projection=lambda o, c: (o, c),
    )
    ee_orders = LiftedLiftedProject(
        LiftedLiftedSelect(joined, lambda pair: pair[1].country == "EE"),
        lambda pair: EEOrder(pair[0].id, pair[0].total),
    )
    view = p.view("ee_orders", DeltaLiftedDistinct(ee_orders))

    p.step(
        {
            orders: [Order(1, 7, 30.0), Order(2, 8, 40.0)],
            customers: [Customer(7, "EE"), Customer(8, "FI")],
        }
    )

    assert view.materialized().inner == {EEOrder(1, 30.0): 1}


def test_program2d_named_relational_set_operators() -> None:
    p = Program2D()
    left: Source[int] = p.source("left")
    right: Source[int] = p.source("right")
    union = p.view("union", DeltaLiftedUnion(left, right))
    except_ = p.view("except", DeltaLiftedExcept(left, right))
    intersect = p.view("intersect", DeltaLiftedIntersect(left, right))

    p.step({left: [1, 2, 3], right: [2, 3, 4]})

    assert union.materialized().inner == {1: 1, 2: 1, 3: 1, 4: 1}
    assert except_.materialized().inner == {1: 1}
    assert intersect.materialized().inner == {2: 1, 3: 1}


def test_program2d_distinct_selection_and_projection() -> None:
    p = Program2D()
    orders: Source[Order] = p.source("orders")
    selected = DeltaLiftedSelectDistinct(orders, lambda o: o.total >= 30.0)
    projected = DeltaLiftedProjectDistinct(selected, lambda o: EEOrder(o.id, o.total))
    view = p.view("large_orders", projected)

    p.step({orders: [Order(1, 7, 30.0), Order(2, 8, 20.0), Order(1, 7, 30.0)]})

    assert view.materialized().inner == {EEOrder(1, 30.0): 1}


def test_program2d_defers_redundant_distincts_to_view_root() -> None:
    p = Program2D()
    orders: Source[Order] = p.source("orders")
    a = DeltaLiftedProjectDistinct(
        DeltaLiftedSelectDistinct(orders, lambda o: o.total >= 30.0),
        lambda o: EEOrder(o.id, o.total),
    )
    b = DeltaLiftedDistinct(DeltaLiftedProjectDistinct(orders, lambda o: EEOrder(o.id, o.total)))
    view = p.view("orders", DeltaLiftedUnion(a, b))

    p.step({orders: [Order(1, 7, 30.0), Order(1, 7, 30.0)]})

    assert _distinct_kernel_count(p) == 1
    assert view.materialized().inner == {EEOrder(1, 30.0): 1}


def test_program2d_uses_map_distinct_dedup_for_projection() -> None:
    p = Program2D()
    rows: Source[KV] = p.source("rows")
    projected = LiftedLiftedProject(DeltaLiftedDistinct(rows), lambda r: r.key)
    view = p.view("keys", DeltaLiftedDistinct(projected))

    p.step({rows: [KV("a", 1), KV("a", 2), KV("a", 2)]})

    assert _distinct_kernel_count(p) == 1
    assert view.materialized().inner == {"a": 1}


def test_program2d_uses_join_distinct_dedup_for_equi_join() -> None:
    p = Program2D()
    left: Source[KV] = p.source("left")
    right: Source[KV] = p.source("right")
    joined = DeltaLiftedDeltaLiftedEquiJoin(
        DeltaLiftedDistinct(left),
        DeltaLiftedDistinct(right),
        left_key=lambda r: r.key,
        right_key=lambda r: r.key,
        projection=lambda l, r: (l.key, l.value, r.value),
    )
    view = p.view("joined", DeltaLiftedDistinct(joined))

    p.step({left: [KV("a", 1), KV("a", 1)], right: [KV("a", 2), KV("a", 2)]})

    assert _distinct_kernel_count(p) == 1
    assert view.materialized().inner == {("a", 1, 2): 1}


def test_program2d_does_not_materialize_distinct_before_subtract_or_aggregate() -> None:
    p = Program2D()
    left: Source[int] = p.source("left")
    right: Source[int] = p.source("right")
    rows: Source[KV] = p.source("rows")

    difference = LiftedLiftedSubtract(DeltaLiftedDistinct(left), DeltaLiftedDistinct(right))
    summed = LiftedLiftedGroupBySum(
        DeltaLiftedDistinct(rows),
        key=lambda r: r.key,
        value=lambda r: r.value,
        output=Agg,
    )
    difference_view = p.view("difference", DeltaLiftedDistinct(difference))
    summed_view = p.view("sum", DeltaLiftedDistinct(summed))

    p.step({left: [1, 2], right: [2, 3], rows: [KV("a", 2), KV("a", 3)]})

    assert _distinct_kernel_count(p) == 2
    assert difference_view.materialized().inner == {1: 1}
    assert summed_view.materialized().inner == {Agg("a", 5): 1}


def test_logical_normalizer_removes_internal_distinct_nodes() -> None:
    p = Program2D()
    rows: Source[KV] = p.source("rows")
    query = DeltaLiftedDistinct(
        LiftedLiftedProject(
            DeltaLiftedUnion(
                DeltaLiftedDistinct(rows),
                DeltaLiftedSelectDistinct(rows, lambda r: r.value > 0),
            ),
            lambda r: r.key,
        )
    )

    normalized = _normalize_expr(query._expr)

    assert normalized.distinct is True
    assert not _contains_expr(normalized.expr, _DistinctExpr)


def test_logical_normalizer_keeps_aggregate_as_zset_operator_not_distinct_barrier() -> None:
    p = Program2D()
    rows: Source[KV] = p.source("rows")
    query = DeltaLiftedDistinct(
        LiftedLiftedGroupBySum(
            DeltaLiftedDistinct(rows),
            key=lambda r: r.key,
            value=lambda r: r.value,
            output=Agg,
        )
    )

    normalized = _normalize_expr(query._expr)

    assert normalized.distinct is True
    assert _contains_expr(normalized.expr, _AggregateExpr)
    assert not _contains_expr(normalized.expr, _DistinctExpr)


def test_program2d_cartesian_and_equi_join_names() -> None:
    p = Program2D()
    orders: Source[Order] = p.source("orders")
    customers: Source[Customer] = p.source("customers")
    cartesian = p.view(
        "cartesian",
        DeltaLiftedDeltaLiftedCartesianProduct(
            orders,
            customers,
            projection=lambda o, c: (o.id, c.id),
        ),
    )
    equi = p.view(
        "equi",
        DeltaLiftedDeltaLiftedEquiJoin(
            orders,
            customers,
            left_key=lambda o: o.customer,
            right_key=lambda c: c.id,
            projection=lambda o, c: (o.id, c.country),
        ),
    )

    p.step({orders: [Order(1, 7, 30.0)], customers: [Customer(7, "EE"), Customer(8, "FI")]})

    assert cartesian.materialized().inner == {(1, 7): 1, (1, 8): 1}
    assert equi.materialized().inner == {(1, "EE"): 1}


def test_program2d_group_by_sum_and_max_are_2d_deltas() -> None:
    p = Program2D()
    rows: Source[KV] = p.source("rows")
    summed = p.view(
        "sum",
        LiftedLiftedGroupBySum(
            rows,
            key=lambda r: r.key,
            value=lambda r: r.value,
            output=Agg,
        ),
    )
    maxed = p.view(
        "max",
        LiftedLiftedGroupByMax(
            rows,
            key=lambda r: r.key,
            value=lambda r: r.value,
            output=Agg,
        ),
    )

    p.step({rows: [KV("a", 2), KV("a", 3), KV("b", 1)]})
    assert summed.materialized().inner == {Agg("a", 5): 1, Agg("b", 1): 1}
    assert maxed.materialized().inner == {Agg("a", 3): 1, Agg("b", 1): 1}

    p.remove(rows, [KV("a", 3)])
    p.step()
    assert summed.materialized().inner == {Agg("a", 2): 1, Agg("b", 1): 1}
    assert maxed.materialized().inner == {Agg("a", 2): 1, Agg("b", 1): 1}


def test_program2d_distinct_handles_retractions_without_reinitializing() -> None:
    p = Program2D(gc=False)
    numbers: Source[int] = p.source("numbers")
    view = p.view("numbers", DeltaLiftedDistinct(numbers))

    p.step({numbers: [1]})
    evaluator_id = id(p.evaluator)
    assert p.evaluator.gc_enabled is False
    assert view.delta().inner == {1: 1}

    p.remove(numbers, [1])
    p.step()

    assert id(p.evaluator) == evaluator_id
    assert view.delta().inner == {1: -1}
    assert view.materialized().inner == {}


def test_program2d_manual_zset_delta_escape_hatch() -> None:
    p = Program2D()
    numbers: Source[int] = p.source("numbers")
    view = p.view("numbers", DeltaLiftedDistinct(numbers))

    p.step({numbers: ZSet({1: 1, 2: 1})})
    p.step({numbers: ZSet({1: -1})})

    assert view.delta().inner == {1: -1}
    assert view.materialized().inner == {2: 1}


def test_program2d_freezes_graph_on_first_step() -> None:
    p = Program2D()
    numbers: Source[int] = p.source("numbers")
    p.view("numbers", numbers)
    p.step({numbers: [1]})

    with pytest.raises(RuntimeError):
        p.source("late")

    with pytest.raises(RuntimeError):
        p.view("late", numbers)
