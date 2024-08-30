# PyDBSP
<div align="center">
<a href="/assets/" />
<img src="/assets/logo.svg" width=400px" />
</a>
</div>

## Introduction - (a subset of) Differential Dataflow for the masses
This library provides an implementation of the [DBSP](https://arxiv.org/pdf/2203.16684) language for incremental streaming
computation. It is a tool **primarily** meant for **research**. See it as the [PyTorch](https://github.com/pytorch/pytorch) of streaming.

It has `zero` dependencies, and is written in pure python.

## What is DBSP? 
DBSP is differential dataflow's less expressive successor. It is a competing theory and framework to other stream processing systems such as Flink
and Spark.

Its value is most easily understood in that it is capable of transforming "batch" possibly-iterative relational queries into "streaming incremental ones". This however
only shows a fraction of the theory's power. 

As an extreme example, you can find a incremental Interpreter for Datalog under `pydbsp.algorithm`. Datalog is a query language that is similar to SQL, with focus
in efficiently supporting recursion. By implementing Datalog interpretation with `dbsp`, we get an interpreter whose queries can both change during runtime __and__ respond
to new data being streamed in.

## Motivating Examples 

There many more examples living in each `test_*.py` file.

### Streaming Pandas with PyDBSP beats batch Pandas 

Let us start with joins. 

```python
from typing import List, Tuple, Set

def regular_join[K, V1, V2](left: Set[Tuple[K, V1]], right: Set[Tuple[K, V2]]) -> List[Tuple[K, V1, V2]]:
    output: List[Tuple[K, V1, V2]] = []
    for left_key, left_value in left:
        for right_key, right_value in right:
            if left_key == right_key:
                output.append((left_key, left_value, right_value))

    return output

employees = {(0, "kristjan"), (1, "mark"), (2, "mike")}
salaries = {(2, "40000"), (0, "38750"), (1, "50000")}

employees_salaries = regular_join(employees, salaries)
print(f"Regular join: {employees_salaries}")
# Regular join: [(1, 'mark', '50000'), (2, 'mike', '40000'), (0, 'kristjan', '38750')]
```

`regular_join` is a straightforward relational `inner join` implementation. You simply loop over two relations, and
then output those that match according to some key.

```python
from pydbsp.zset import ZSet
from pydbsp.zset.functions.bilinear import join

employees_zset = ZSet({k: 1 for k in employees})
salaries_zset = ZSet({k: 1 for k in salaries})
employees_salaries_zset = join(
    employees_zset,
    salaries_zset,
    lambda left, right: left[0] == right[0],
    lambda left, right: (left[0], left[1], right[1]),
)
print(f"ZSet join: {employees_salaries_zset}")
# ZSet join: {(1, 'mark', '50000'): 1, (2, 'mike', '40000'): 1, (0, 'kristjan', '38750'): 1}
```

The core of `dbsp` is a simple, but scary-named, mathematical construct, the Abelian group. A group is
a set with associated `+` and `-` operations of __a certain kind__.

`ZSet`'s are a group of special interest to us. They are exactly the same as sets, except **each element** is associated 
with a **weight**. When one adds two of these, the result is the union of both sets with the weights of identical elements 
summed. Negation is as straightforward as it sounds. You just flip the sign on each element's weight.

Notice how it is possible to model regular sets, bags and set updates with them. A set is a `ZSet` where each weight is exactly `1`, a bag 
where it can be more than `1`, and an "update" is one where it is either `1` or `-1`. 

The gist of `dbsp` is that certain kinds of functions, those called `linear`, can be efficiently incrementalized. Many useful
functions are `linear`. For instance, `regular_join` is. 

A function is `linear`, or `bilinear` if it has two arguments, if it distributes over addition. For some function `f`, it is
linear if for each `a`, `b`, `c` of the same group, it holds: `f(a + b) = f(a) + f(b)`.

This might be a tad abstract, but let's garner some intuition by looking at it from the join perspective and how it gives a blueprint
to make it incremental.

The join of the running example has been, taking `E` as the set of employees and `S` of salary: `E ⨝ S`

If we call `ΔE`and `ΔS` sets of updates, the "batch" way of evaluating this query under an update would be: `(E + ΔE) ⨝ (S + ΔS)` 

Since it __distributes over addition__, we could also evaluate it __incrementally__: `(ΔE) ⨝ (ΔS) + (E) ⨝ (ΔS) + (ΔE) ⨝ (S)`.

When done that way, we do three joins instead of one, but notice how each join has at least one side of updates. This makes it much more
efficient, since we effectively shift the lower bound to go from "all data" to "the update". 


```python

from pydbsp.zset import ZSetAddition
from pydbsp.stream import Stream, StreamHandle
from pydbsp.stream.operators.linear import Integrate
from pydbsp.zset.operators.bilinear import LiftedJoin

group = ZSetAddition()
employees_stream = Stream(group)
employees_stream_handle = StreamHandle(lambda: employees_stream)
employees_stream.send(employees_zset)

salaries_stream = Stream(group)
salaries_stream_handle = StreamHandle(lambda: salaries_stream)
salaries_stream.send(salaries_zset)

join_cmp = lambda left, right: left[0] == right[0]
join_projection = lambda left, right: (left[0], left[1], right[1])

integrated_employees = Integrate(employees_stream_handle)
integrated_salaries = Integrate(salaries_stream_handle)
stream_join = LiftedJoin(
    integrated_employees.output_handle(),
    integrated_salaries.output_handle(),
    join_cmp,
    join_projection,
)
integrated_employees.step()
integrated_salaries.step()
stream_join.step()
print(f"ZSet stream join: {stream_join.output().latest()}")
# ZSet stream join: {(1, 'mark', '50000'): 1, (2, 'mike', '40000'): 1, (0, 'kristjan', '38750'): 1}
```

Now, streams. A stream is an infinite list. We say that to `lift` some function is to apply it element-wise to some stream. `LiftedJoin` in 
the example is `join` applied element-wise to two `ZSet` streams. The result of the stream join is then predictably the same as the regular `ZSet` join.

`Integrate` is an operator, a function whose input and output are streams, that at each time step contains the cumulative sum of all values so far.

The stream join that is depicted then yields `(E + ΔE) ⨝ (S + ΔS)` at each timestep.

```python
from pydbsp.stream.operators.bilinear import Incrementalize2

incremental_stream_join = Incrementalize2(
    employees_stream_handle,
    salaries_stream_handle,
    lambda left, right: join(left, right, join_cmp, join_projection),
    group,
)
incremental_stream_join.step()
print(f"Incremental ZSet stream join: {incremental_stream_join.output().latest()}")
# Incremental ZSet stream join: {(0, 'kristjan', '38750'): 1, (1, 'mark', '50000'): 1, (2, 'mike', '40000'): 1}
```

We can immediately make it "incremental" just by using the `Incrementalize2` operator. One of its arguments is the `bilinear` function to incrementalize, which
in our case is the `ZSet` join, that then automatically assembles: `(ΔE) ⨝ (ΔS) + (E) ⨝ (ΔS) + (ΔE) ⨝ (S)`.

```python
employees_stream.send(ZSet({(2, "mike"): -1}))
incremental_stream_join.step()
print(f"Incremental ZSet stream join update: {incremental_stream_join.output().latest()}")
# Incremental ZSet stream join update: {(2, 'mike', '40000'): -1}
```

Modern streaming systems often handle deletion poorly, and in many times they just don't. By using `dbsp` however we get this for free. If we send in a set with elements 
that have negative weight, this weight will "propagate" forward. In this example, by retracting `mike` we also retract the result of the join.

Cool! we went from batch all the way to incremental stream processing with very few lines of code.

```python
from pydbsp.indexed_zset.functions.bilinear import join_with_index
from pydbsp.indexed_zset.operators.linear import LiftedIndex

indexer = lambda x: x[0]
index_employees = LiftedIndex(employees_stream_handle, indexer)
index_salaries = LiftedIndex(salaries_stream_handle, indexer)
incremental_sort_merge_join = Incrementalize2(index_employees.output_handle(), index_salaries.output_handle(), lambda l, r: join_with_index(l, r, join_projection), group)
index_employees.step()
index_salaries.step()
incremental_sort_merge_join.step()
print(f"Incremental indexed ZSet stream join: {incremental_sort_merge_join.output().latest()}")
# Incremental indexed ZSet stream join: {(0, 'kristjan', '38750'): 1, (1, 'mark', '50000'): 1, (2, 'mike', '40000'): 1}
```

```python
index_employees.step()
incremental_sort_merge_join.step()
print(f"Incremental ZSet stream join update: {incremental_sort_merge_join.output().latest()}")
# Incremental ZSet stream join update: {(2, 'mike', '40000'): -1}
```

There are multiple ways to implement joins. The three most common kinds are:
1. Hash 
2. Nested loop
3. Sort-merge

Adding b-tree indexes to a database table makes 3., often the most efficient, possible. Our `regular_join` is a nested loop join. Are we also able to somehow add "indexes" to our
streams? Yes! b-tree Indexing is linear. The `LiftedIndex` operator "indexes" both `employees` and `salaries` sets by their first column. 

```python
from random import randrange

names = ("kristjan", "mark", "mike")
max_pay = 100000
fake_data = [((i, names[randrange(len(names))] + str(i)), (i, randrange(max_pay))) for i in range(3, 10003)]
batch_size = 500
fake_data_batches = [fake_data[i : i + batch_size] for i in range(0, len(fake_data), batch_size)]

for batch in fake_data_batches:
    employees_stream.send(ZSet({employee: 1 for employee, _ in batch}))
    salaries_stream.send(ZSet({salary: 1 for _, salary in batch}))

steps_to_take = len(fake_data_batches)
```

We have implemented many variations of a streaming join:
1. Batch
2. Incremental
3. Incremental with indexing

Let's add one more variant, this time using pandas.

To compare all of these we will run a simple benchmark using not a lot of data. As the snippet shows, there will be `20` batches with each containing `500` 
employees and salaries.

```python
from tqdm.notebook import tqdm
from time import time

time_start = time()
measurements = []
for _ in tqdm(range(steps_to_take)):
    local_time = time()
    integrated_employees.step()
    integrated_salaries.step()
    stream_join.step()
    measurements.append(time() - local_time)
print(f"Time taken - on demand: {time() - time_start}s")
# Time taken - on demand: 20.57329797744751s
```

Computing all 20 batches with a regular stream join took a whopping...20 seconds. Ouch. That is very slow.

I have an excuse however. The goal of the baseline `ZSet` implementation is to be simple to inspect and debug. 

```python
import pandas as pd

time_start = time()
pandas_measurements = []
employees_union_df = pd.DataFrame(columns=['id', 'name'])
salaries_union_df = pd.DataFrame(columns=['id', 'salary'])

for step in tqdm(range(steps_to_take)):
    local_time = time()
    employees_batch_df = pd.DataFrame([ employee for employee, _ in fake_data_batches[step] ], columns=['id', 'name'])
    employees_union_df = pd.concat([employees_union_df, employees_batch_df], ignore_index=True)

    salaries_batch_df = pd.DataFrame([ salary for _, salary in fake_data_batches[step] ], columns=['id', 'salary'])
    salaries_union_df = pd.concat([salaries_union_df, salaries_batch_df], ignore_index=True)
    
    employees_x_salaries = pd.merge(employees_union_df, salaries_union_df, on=['id'], how='inner')
    pandas_measurements.append(time() - local_time)

print(f"Time taken - on demand with pandas: {time() - time_start}s")
# Time taken - on demand with pandas: 0.032193899154663086s
```

Well, pandas blew it out of the water taking a satanic `666x` less time.

```python
time_start = time()
incremental_measurements = []
for _ in tqdm(range(steps_to_take)):
    local_time = time()
    incremental_stream_join.step()
    incremental_measurements.append(time() - local_time)
print(f"Time taken - incremental: {time() - time_start}s")
# Time taken - incremental: 2.9529590606689453s
```

With the `incremental` variant of the stream join we get a `6.66x` improvement, still `100x` slower than pandas.

```python
time_start = time()
incremental_with_index_measurements = []
for _ in tqdm(range(steps_to_take)):
    local_time = time()
    index_employees.step()
    index_salaries.step()
    incremental_sort_merge_join.step()
    incremental_with_index_measurements.append(time() - local_time)
print(f"Time taken - incremental with index: {time() - time_start}s")
# Time taken - incremental with index: 0.16031384468078613s
```

<div align="center">
<a href="/assets/" />
<img src="/assets/variants.png" width=400px" />
</a>
</div>


That's it folks, indexing **does** make a difference. That's already more than 100 times faster than the original batch solution, with
almost no change. 

Now we are only down to being `5x` slower than pandas (this indeed hurts to spell out) using pure python.

```python
batch_size = 20000

lots_of_fake_data = [((i, names[randrange(len(names))] + str(i)), (i, randrange(max_pay))) for i in range(3000000)]
lots_of_fake_data_batches = [lots_of_fake_data[i : i + batch_size] for i in range(0, len(lots_of_fake_data), batch_size)]

new_pandas_measurements = []
employees_union_df = pd.DataFrame(columns=['id', 'name'])
salaries_union_df = pd.DataFrame(columns=['id', 'salary'])
new_steps_to_take = len(lots_of_fake_data_batches)
results_pandas = [] 

time_start = time()
for step in tqdm(range(new_steps_to_take)):
    local_time = time()
    employees_batch_df = pd.DataFrame([ employee for employee, _ in lots_of_fake_data_batches[step] ], columns=['id', 'name'])
    employees_union_df = pd.concat([employees_union_df, employees_batch_df], ignore_index=True)

    salaries_batch_df = pd.DataFrame([ salary for _, salary in lots_of_fake_data_batches[step] ], columns=['id', 'salary'])
    salaries_union_df = pd.concat([salaries_union_df, salaries_batch_df], ignore_index=True)
    
    employees_x_salaries = pd.merge(employees_union_df, salaries_union_df, on='id', how='inner')
    results_pandas.append(employees_x_salaries)
    new_pandas_measurements.append(time() - local_time)

print(f"Time taken - on demand with pandas: {time() - time_start}s")
# Time taken - on demand with pandas: 24.699557065963745s
```

Let's go big. At what point does pandas start to struggle? If we shift the batch up to `20000` new employees and salaries, and push 150 of these, pandas
takes around 24 seconds to churn through.

Now, for what you've been waiting for. Could we leverage `pydbsp` to speed this up?

```python
from pydbsp.core import AbelianGroupOperation

class ImmutableDataframeZSet:
    def __init__(self, df: pd.DataFrame) -> None:
        if 'weight' not in df.columns:
            raise ValueError("DataFrame must have a 'weight' column")
        self.inner: List[pd.DataFrame] = [df[df['weight'] != 0]]

    def __repr__(self) -> str:
        return self.inner.__repr__()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ImmutableDataframeZSet):
            return False
        if len(self.inner) != len(other.inner):
            return False
        return all(df1 is df2 for df1, df2 in zip(self.inner, other.inner))
```

The first step is to define a pandas-backed `ZSet`. This is quite straightforward. Let's consider all pandas dataframes with a `weight` column to be `ZSet`s.

Next, since we are only interested in linear or bilinear functions, let's **never** concatenate dataframes. This is okay because once again, they distribute
over addition.

```python
class ImmutableDataframeZSetAddition(AbelianGroupOperation[ImmutableDataframeZSet]):
    def add(self, a: ImmutableDataframeZSet, b: ImmutableDataframeZSet) -> ImmutableDataframeZSet:
        result = ImmutableDataframeZSet(pd.DataFrame(columns=a.inner[0].columns))
        result.inner = a.inner + b.inner
        return result

    def neg(self, a: ImmutableDataframeZSet) -> ImmutableDataframeZSet:
        result = ImmutableDataframeZSet(pd.DataFrame(columns=a.inner[0].columns))
        result.inner = [df.assign(weight=lambda x: -x.weight) for df in a.inner]
        return result

    def identity(self) -> ImmutableDataframeZSet:
        return ImmutableDataframeZSet(pd.DataFrame(columns=['weight']))

```

Addition is simple and very lightweight. We can "add" two pandas-backed `ZSet`s by contatenating their dataframe lists. That's it.

```python
def join_dfs(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    join_columns: List[str]
):
    if left_df.empty or right_df.empty:
        return pd.DataFrame()
    
    joined = pd.merge(left_df, right_df, on=join_columns, how='inner', suffixes=('_left', '_right'))
    joined['weight'] = joined['weight_left'] * joined['weight_right']
    joined = joined.drop(['weight_left', 'weight_right'], axis=1)

    return joined
    
def immutable_dataframe_zset_join(
    left_zset: ImmutableDataframeZSet,
    right_zset: ImmutableDataframeZSet,
    join_columns: List[str]
) -> ImmutableDataframeZSet:
    join_tasks = [(left_df, right_df, join_columns) 
                  for left_df in left_zset.inner 
                  for right_df in right_zset.inner]

    result_dfs = [ join_dfs(left, right, join_columns) for left, right, join_columns in join_tasks ]

    non_empty_dfs = [df for df in result_dfs if not df.empty]

    if not non_empty_dfs:
        return immutable_df_abelian_group.identity()

    result = ImmutableDataframeZSet(pd.DataFrame(columns=non_empty_dfs[0].columns))
    result.inner = non_empty_dfs
    return result
```

Now here is where the speedup is visible.

To join two pandas-backed `ZSet`s, we can simply join **every single** dataframe from the left side, with the right side. Remember, join distributes over addition!

```python
employees_with_weight = [ employee + (1,) for employee in employees ]
salaries_with_weight = [ salary + (1,) for salary in salaries ]  

employees_pandas_zset = ImmutableDataframeZSet(pd.DataFrame(employees_with_weight, columns=['id', 'name', 'weight']))
salaries_pandas_zset = ImmutableDataframeZSet(pd.DataFrame(salaries_with_weight , columns=['id', 'salary', 'weight']))

print(immutable_dataframe_zset_join(employees_pandas_zset, salaries_pandas_zset, 'id'))
# [   id      name salary  weight
# 0   0  kristjan  38750       1
# 1   1      mark  50000       1
# 2   2      mike  40000       1]
```

Seems like it works!

```python
employees_dfs_stream = Stream(immutable_df_abelian_group)
employees_dfs_stream_handle = StreamHandle(lambda: employees_dfs_stream)

salaries_dfs_stream = Stream(immutable_df_abelian_group)
salaries_dfs_stream_handle = StreamHandle(lambda: salaries_dfs_stream)

incremental_pandas_join = Incrementalize2(employees_dfs_stream_handle, salaries_dfs_stream_handle, lambda l, r: immutable_dataframe_zset_join(l, r, ['id']), immutable_df_abelian_group)
incremental_pandas_measurements = []
time_start = time()
for step in tqdm(range(new_steps_to_take)):
    local_time = time()
    employees_batch_df = pd.DataFrame([ employee + (1,) for employee, _ in lots_of_fake_data_batches[step] ], columns=['id', 'name', 'weight'])
    employees_dfs_stream.send(ImmutableDataframeZSet(employees_batch_df))
    
    salaries_batch_df = pd.DataFrame([ salary + (1,) for _, salary in lots_of_fake_data_batches[step] ], columns=['id', 'salary', 'weight'])
    salaries_dfs_stream.send(ImmutableDataframeZSet(salaries_batch_df))
        
    incremental_pandas_join.step()
    incremental_pandas_measurements.append(time() - local_time)
print(f"Time taken - incremental: {time() - time_start}s")
# Time taken - incremental: 14.954236268997192s
```

<div align="center">
<a href="/assets/" />
<img src="/assets/pandas_variants.png" width=400px" />
</a>
</div>

Amazing! we **did** indeed get an almost `2x` speedup. Definitely less dramatic than the previous ones, but a significant one nonetheless. 
