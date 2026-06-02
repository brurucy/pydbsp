# PyDBSP
<div align="center">
<img src="https://raw.githubusercontent.com/brurucy/pydbsp/master/assets/logo.svg" width="400px" />
</div>

## Introduction - (a subset of) Differential Dataflow for the masses
This library provides an implementation of the [DBSP](https://arxiv.org/pdf/2203.16684) language for incremental streaming
computations. It is a tool **primarily** meant for **research**. See it as the [PyTorch](https://github.com/pytorch/pytorch) of streaming.

As of v2.0.0 it has **zero runtime dependencies** and is written in pure Python.

[**Here**](https://github.com/brurucy/dbsp-from-scratch/blob/master/dbsp_paper_walkthrough_implementation.ipynb) you can find a single-notebook implementation of almost everything in
the [DBSP](https://arxiv.org/pdf/2203.16684) paper. It mirrors what is in this library in an accessible way, and with more examples.

## What is DBSP?
DBSP is differential dataflow's less expressive successor. It is a competing theory and framework to other stream processing systems
such as Flink and Spark.

Its value is most easily understood in that it is capable of transforming "batch" possibly-iterative relational queries
into "streaming incremental ones". This however only conveys a fraction of the theory's power.

As an extreme example, this library ships an incremental interpreter for **stratified Datalog with negation** (`pydbsp.datalog_stratified.IncrementalDatalogStratified`). Datalog is a query language similar to SQL, focused on efficiently supporting recursion. By implementing Datalog interpretation with DBSP, we get an interpreter whose queries can both change during runtime __and__ respond to new data being streamed in.

## Examples

### A small banking pipeline

Given the following SQL views over a `transactions(id, from_account, to_account, amount)` table:

```sql
create view credits as
  select to_account as account, sum(amount) as credits
  from transactions group by to_account;
create view debits as
  select from_account as account, sum(amount) as debits
  from transactions group by from_account;
create view balance as
  select credits.account, credits - debits as balance
  from credits inner join debits on credits.account = debits.account;
create materialized view total as
  select sum(balance) from balance;
```

The PyDBSP equivalent wires the same dataflow incrementally. The `credits`, `debits`, and `total` views use `DeltaLiftedDeltaLiftedGroupBy`, the incremental `GROUP BY ... AGGREGATE`, so each push re-aggregates only the accounts it touched and emits the changed `(account, total)` pairs directly. Transactions arrive one per outer tick, and after every push we read out the latest total. Records are `(id, from_account, to_account, amount)` tuples:

```python
from pydbsp.circuit import Circuit
from pydbsp.compute import ComputeCtx
from pydbsp.core import Antichain, dbsp_time
from pydbsp.evaluate import Evaluator
from pydbsp.indexed_relational_operators import (
    DeltaLiftedDeltaLiftedGroupBy, IndexedDeltaLiftedDeltaLiftedJoin,
    LiftIndex, LiftLiftIndex,
)
from pydbsp.indexed_zset import IndexedZSetAddition
from pydbsp.operator import Input, Integrate, Lift1, LiftStreamIntroduction
from pydbsp.storage import DictStorage
from pydbsp.zset import ZSet, ZSetAddition

g_txn = ZSetAddition()   # transactions (id, from, to, amount)
g_kv = ZSetAddition()    # (account, amount) pairs
g_by_to = IndexedZSetAddition(g_txn, lambda r: r[2])
g_by_from = IndexedZSetAddition(g_txn, lambda r: r[1])
g_idx = IndexedZSetAddition(g_kv, lambda kv: kv[0])
g_total = ZSetAddition()

e = Evaluator(
    circuit=Circuit(),
    storage=DictStorage(),
    ctx=ComputeCtx(lattice=dbsp_time(2)),
    group=g_txn,
)
src = Input(frontier=Antichain(dbsp_time(1))).connect(e.circuit, ())
src_2d = LiftStreamIntroduction(group=g_txn).connect(e.circuit, (src,))

# Incremental GROUP BY: emit (account, total) deltas directly, with no
# Integrate/Differentiate scaffolding around a full re-aggregation.
sum_amount = lambda items: sum(r[3] * w for r, w in items)
credits = DeltaLiftedDeltaLiftedGroupBy(
    aggregate=sum_amount, group=g_by_to, out_group=g_kv,
).connect(e.circuit, (LiftLiftIndex(indexer=lambda r: r[2]).connect(e.circuit, (src_2d,)),))
debits = DeltaLiftedDeltaLiftedGroupBy(
    aggregate=sum_amount, group=g_by_from, out_group=g_kv,
).connect(e.circuit, (LiftLiftIndex(indexer=lambda r: r[1]).connect(e.circuit, (src_2d,)),))

balance_delta = IndexedDeltaLiftedDeltaLiftedJoin(
    proj=lambda k, c, d: (k, c[1] - d[1]),
    group_a=g_idx, group_b=g_idx, out_group=g_kv,
).connect(
    e.circuit,
    (
        LiftIndex(indexer=lambda kv: kv[0]).connect(e.circuit, (credits,)),
        LiftIndex(indexer=lambda kv: kv[0]).connect(e.circuit, (debits,)),
    ),
)
balance = Integrate(group=g_kv).connect(e.circuit, (balance_delta,))

sum_balance = lambda items: sum(b[1] * w for b, w in items)
total_delta = DeltaLiftedDeltaLiftedGroupBy(
    aggregate=sum_balance,
    group=IndexedZSetAddition(g_kv, lambda _kv: ()), out_group=g_total,
).connect(e.circuit, (LiftLiftIndex(indexer=lambda _kv: ()).connect(e.circuit, (balance_delta,)),))
total_cum = Integrate(group=g_total).connect(e.circuit, (total_delta,))
total = Lift1(
    f=lambda z: next((c for (_k, c), _w in z.inner.items()), 0),
).connect(e.circuit, (total_cum,))

# Stream one transaction per outer tick.
for tick, txn in enumerate([
    (0, 1, 2, 50),   # cumulative total after this tick: 0
    (1, 2, 3, 30),   # cumulative total after this tick: 20
    (2, 3, 1, 20),   # cumulative total after this tick: 0
]):
    e.push(src, ZSet({txn: 1}))
    print(f"tick {tick}: total = {e.read(total, (tick, 0))}")
print("final balance:", dict(sorted(e.read(balance, (2, 0)).inner.items())))
#   → {(1, -30): 1, (2, 20): 1, (3, 10): 1}
```

Every credit is some account's debit, so `total` nets to zero whenever the inner join of credits and debits has caught up with every account. Early ticks may report a non-zero total while some account is still missing from one side of the join.

### Theory references

* [Lean declaration index](https://github.com/brurucy/pydbsp/blob/master/dbsp-theory-summarised.md) — auto-generated inventory of the 527 top-level `def` / `lemma` / `theorem` entries in the DBSP Lean formalisation, with a one-line signature for each. Useful as a grep target rather than an editorial reference.
* [Progress algebra](https://github.com/brurucy/pydbsp/blob/master/progress-algebra-summarised.md) — the antichain machinery PyDBSP is built on: the time lattice, down-sets, antichains, the per-axis `shift` / `retreat` / `drop_axis` / `insert_axis` primitives, settled frontiers, and the forward / backward propagation walks.
* [Implementation of the DBSP Paper](https://github.com/brurucy/dbsp-from-scratch) — single-notebook port of almost everything in the paper.

### Papers

* [Incremental Evaluation of Dynamic Datalog Programs as a Higher-order DBSP Program](https://ceur-ws.org/Vol-3801/paper1.pdf) — Rucy Carneiro Alves de Lima, Apinis, Kramer, and Micinski. *5th International Workshop on the Resurgence of Datalog in Academia and Industry (Datalog-2.0 2024)*. The first paper to use PyDBSP.

### Blogposts (PyDBSP v0.6.0)

* [Streaming Pandas on the GPU](https://www.feldera.com/blog/gpu-stream-dbsp)

### Notebooks

* [Quickstart](https://github.com/brurucy/pydbsp/blob/master/notebooks/quickstart.ipynb) — six primitives tour with two canonical pipelines.
* [Tutorial](https://github.com/brurucy/pydbsp/blob/master/notebooks/readme.ipynb) — six-section tour of the public API: Z-sets, the evaluator, the doubly-incremental DLD join, sort-merge indexing, transitive closure, and Datalog.
* [WordCount](https://github.com/brurucy/pydbsp/blob/master/notebooks/wordcount.ipynb) — the Kafka Streams WordCount topology as a DBSP circuit, with weight-encoded counts, the `(word, count)` changelog via `DeltaLiftedDeltaLiftedGroupBy`, and retraction-aware corrections.
* [SQL operator walkthrough](https://github.com/brurucy/pydbsp/blob/master/notebooks/sql.ipynb) — incremental forms of the SQL operators from §4.2 of the DBSP paper.
* [Datalog](https://github.com/brurucy/pydbsp/blob/master/notebooks/datalog.ipynb) — `IndexedIncrementalDatalogBody` on a transitive-closure program, batched vs. drip-fed.
* [Stratified Datalog with negation](https://github.com/brurucy/pydbsp/blob/master/notebooks/stratified_negation.ipynb) — `IncrementalDatalogStratified` on two negation programs (transitive complement; 4-cycle without an overlapping 3-cycle).
* [RDFS materialization](https://github.com/brurucy/pydbsp/blob/master/notebooks/rdfs.ipynb) — LUBM1 through `IndexedIncrementalRDFSBody` cross-checked against a Datalog re-encoding.
* [Benchmarks](https://github.com/brurucy/pydbsp/blob/master/notebooks/benchmark.ipynb) — sort-merge-indexed reachability and Datalog TC across the bundled graphs.
* [Internals dissection](https://github.com/brurucy/pydbsp/blob/master/notebooks/pydbsp_internals_dissection.ipynb) — eight cells tracing the jamie SQL aggregate pipeline at progressively finer resolution.

### Tests

There are many examples living in each `tests/test_*.py` file.
