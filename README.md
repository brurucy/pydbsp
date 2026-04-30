# PyDBSP
<div align="center">
<a href="/assets/" />
<img src="/assets/logo.svg" width=400px" />
</a>
</div>

## Introduction - (a subset of) Differential Dataflow for the masses
This library provides an implementation of the [DBSP](https://arxiv.org/pdf/2203.16684) language for incremental streaming
computations. It started as a research-first reference implementation — see it as the [PyTorch](https://github.com/pytorch/pytorch) of streaming — and has since grown into something usable: a schedule-driven evaluator with explicit per-operator dependencies, antichain-guided garbage collection, and per-layer parallelism that scales on free-threaded Python (3.14t).

It has `zero` dependencies, and is written in pure python.

[**Here**](https://github.com/brurucy/dbsp-from-scratch/blob/master/dbsp_paper_walkthrough_implementation.ipynb) you can find a single-notebook implementation of almost everything in 
the [DBSP](https://arxiv.org/pdf/2203.16684) paper. It mirrors what is in this library in an accessible way, and with more examples.

## What is DBSP? 
DBSP is differential dataflow's less expressive successor. It is a competing theory and framework to other stream processing systems 
such as Flink and Spark.

Its value is most easily understood in that it is capable of transforming "batch" possibly-iterative relational queries 
into "streaming incremental ones". This however only conveys a fraction of the theory's power. 

As an extreme example, you can find a incremental Interpreter for Datalog under `pydbsp.algorithm`. Datalog is a query language that is 
similar to SQL, with focus in efficiently supporting recursion. By implementing Datalog interpretation with `dbsp`, we get an interpreter
whose queries can both change during runtime __and__ respond to new data being streamed in.

## Examples 

### Python API

The default user-facing path owns outer ticks and materialization for
you:

```python
from pydbsp import Datalog, V, atom, fact, facts, rule, rules

X, Y, Z = V("X"), V("Y"), V("Z")
program = rules(
    rule(atom("tc", X, Y), atom("e", X, Y)),
    rule(atom("tc", X, Z), atom("tc", X, Y), atom("e", Y, Z)),
)

db = Datalog(indexed=True, parallelism=1)
db.step(
    facts=facts(fact("e", 0, 1), fact("e", 1, 2), fact("e", 2, 3)),
    program=program,
)

print(db.relation("tc").inner)
```

For programs with stratified negation, use `StratifiedDatalog` — it
runs on the 3-D `(outer, stratum, inner)` time lattice so each
stratum's fixpoint is delayed into its successor. Note: the 3-D path
is currently unrefined and roughly an order of magnitude slower than
2-D `Datalog` on equivalent workloads, with parallel scaling capped
near 1.4×. It is the only path to full multi-stratum negation today;
expect a perf cliff vs `Datalog` (which handles the semipositive
case):

```python
from pydbsp import StratifiedDatalog, V, atom, fact, facts, not_, rule, rules

X = V("X")
program = rules(
    rule(atom("alive", X), atom("person", X), not_(atom("dead", X))),
)
db = StratifiedDatalog()
db.step(
    facts=facts(fact("person", "alice"), fact("person", "bob"), fact("dead", "bob")),
    program=program,
)
print(db.relation("alive").inner)  # {('alice',): 1}
```

`Reachability` and `RDFS` follow the same shape, all accepting
`parallelism=N` — the schedule-driven evaluator dispatches each
P-antichain layer over a thread pool, which scales on free-threaded
Python (3.14t / `PYTHON_GIL=0`).

The lower-level circuit API remains available under
`pydbsp.algorithms.*` for experiments that need direct access to
inputs, feedback state, evaluators, and saturation drivers.

For non-iterative streaming relational queries, use the typed 2-D DBSP
interface. It keeps DBSP operator names visible while hiding lattice and
evaluator boilerplate:

```python
from typing import NamedTuple

from pydbsp import (
    DeltaLiftedDeltaLiftedSortMergeJoin,
    DeltaLiftedDistinct,
    Program2D,
    Source,
)


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


p = Program2D(gc=True)
orders: Source[Order] = p.source("orders")
customers: Source[Customer] = p.source("customers")

ee_orders = DeltaLiftedDeltaLiftedSortMergeJoin(
    orders,
    customers,
    left_key=lambda o: o.customer,
    right_key=lambda c: c.id,
    projection=lambda o, _c: EEOrder(o.id, o.total),
)

view = p.view("ee_orders", DeltaLiftedDistinct(ee_orders))
p.step({orders: [Order(1, 7, 30.0)], customers: [Customer(7, "EE")]})

print(view.delta().inner)
print(view.materialized().inner)
```

Typed queries are compiled as logical plans when a view is registered.
`DeltaLiftedDistinct` is normalized as a final result constraint: nested
or intermediate distincts are removed and at most one physical DBSP
distinct is emitted at the view root. This matches the DBSP distinct
deduplication laws and avoids redundant distinct circuits.

Aggregates use Z-set semantics directly. `LiftedLiftedAggregate` receives
the cumulative `ZSet` for its input and returns a `ZSet` delta relation;
`LiftedLiftedGroupBySum` and `LiftedLiftedGroupByMax` are convenience
wrappers built on that primitive. If an aggregate wants set-input
semantics, it should interpret/clamp weights inside the aggregate
function explicitly.

### Paper walkthroughs

* [Implementation of the DBSP Paper](https://github.com/brurucy/dbsp-from-scratch)

### Blogposts

* [Streaming Pandas on the GPU](https://www.feldera.com/blog/gpu-stream-dbsp)

### Notebooks

* [**Quickstart** — the 90% path in two cells](notebooks/quickstart.ipynb)
* [Progress tracking, visually — lattice, antichain, frontier, GC](notebooks/progress_tracking.ipynb)
* [Graph Reachability](notebooks/benchmark.ipynb)
* [Datalog Interpretation](notebooks/datalog.ipynb)
* [Stratified Datalog Interpretation](notebooks/stratified_negation.ipynb)
* [Not-interpreted Datalog](notebooks/rdfs.ipynb)
* [Streaming Pandas](notebooks/readme.ipynb)
* [SQL Operators in DBSP](notebooks/sql.ipynb)
* [Pure-MLX GPU backend experiment](notebooks/readme_gpu.ipynb)

### Tests

There many examples living in each `test/test_*.py` file.
