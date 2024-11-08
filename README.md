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

### Paper walkthroughs

* [Implementation of the DBSP Paper](https://github.com/brurucy/dbsp-from-scratch)

### Blogposts

* [Streaming Pandas on the GPU](https://www.feldera.com/blog/gpu-stream-dbsp)

### Notebooks

* [Graph Reachability](notebooks/benchmark.ipynb)
* [Datalog Interpretation](notebooks/datalog.ipynb)
* [Not-interpreted Datalog](notebooks/rdfs.ipynb)
* [Streaming Pandas](notebooks/readme.ipynb)
* [Streaming Pandas on the GPU](notebooks/readme_gpu.ipynb)

### Tests

There many examples living in each `test/test_*.py` file.
