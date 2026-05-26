"""pydbsp. Incremental relational stream processing on a flat product
lattice of natural-number axes.

The top level is intentionally bare. Import what you need from the
submodules:

* :mod:`pydbsp.core`. Antichain / lattice primitives, group protocol.
* :mod:`pydbsp.zset`. Z-sets and the linear / bilinear / binary value
  functions (``join``, ``project``, ``select``, ``H``).
* :mod:`pydbsp.indexed_zset`. Indexed Z-sets and ``sort_merge_join``.
* :mod:`pydbsp.progress`. Antichain progress shapes + forward /
  backward propagation.
* :mod:`pydbsp.compute`. Value rules per node.
* :mod:`pydbsp.storage`. Per-``(NodeId, Time)`` storage protocol +
  ``DictStorage`` default.
* :mod:`pydbsp.operator`. DBSP operators that pair progress + compute
  via :class:`pydbsp.circuit.Circuit`.
* :mod:`pydbsp.circuit`. Circuit arena.
* :mod:`pydbsp.evaluate`. Evaluator over (Circuit, Storage).
* :mod:`pydbsp.relational_operators`. Z-set relational operators on
  top of the operator layer.
* :mod:`pydbsp.indexed_relational_operators`. Indexed-Z-set
  relational operators (sort-merge join, LiftIndex, LiftGroupBy).
* :mod:`pydbsp.datalog`. Datalog value-layer terms (Rewrite, unify,
  sig, dir, plus the indexed extensions).
* :mod:`pydbsp.rdfs`. RDFS body operator (six rules baked in).
"""
