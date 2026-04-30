# PyDBSP Local DBSP Theory Notes

This file inventories every top-level Lean `def`, `lemma`, and `theorem` I found in the DBSP formalization, including `private`, `protected`, and `noncomputable` declarations. Proofs are intentionally omitted; each item is described only by role and statement.

Total declarations covered: **527**.

## `aggregation.lean`

Aggregations from Z-sets down to scalar summaries, plus singleton-Z-set wrappers that keep those summaries inside the relational algebra pipeline.

Declarations in this module: **10**.

### `count`

- Kind: `def`
- Source: `src/aggregation.lean:11`
- Signature: `def count (m: Z[a]) : ℤ`
- Description: Defines the integer-valued count of a Z-set by summing all multiplicities in its support.

### `count'`

- Kind: `def`
- Source: `src/aggregation.lean:14`
- Signature: `def count' : Z[a] → Z[ℤ]`
- Description: Wraps `count` into a singleton Z-set so aggregation results can remain inside the Z-set pipeline.

### `count_ok`

- Kind: `theorem`
- Source: `src/aggregation.lean:16`
- Signature: `theorem count_ok (s: finset a) : count (zset.from_set s) = s.card`
- Description: Shows that `count` matches its intended semantics.

### `count_linear`

- Kind: `theorem`
- Source: `src/aggregation.lean:25`
- Signature: `theorem count_linear (m1 m2: Z[a]) : count (m1 + m2) = count m1 + count m2`
- Description: Shows that `count` is linear.

### `count'_ok`

- Kind: `theorem`
- Source: `src/aggregation.lean:45`
- Signature: `theorem count'_ok (s: finset a) : zset.to_set (count' (zset.from_set s)) = {s.card}`
- Description: Shows that `count'` matches its intended semantics.

### `sum`

- Kind: `protected def`
- Source: `src/aggregation.lean:57`
- Signature: `protected def sum (m: Z[ℚ]) : ℚ`
- Description: Defines the weighted numeric sum of a rational Z-set.

### `sum'`

- Kind: `def`
- Source: `src/aggregation.lean:60`
- Signature: `def sum' : Z[ℚ] → Z[ℚ]`
- Description: Wraps `zset.sum` into a singleton Z-set result.

### `sum_ok`

- Kind: `theorem`
- Source: `src/aggregation.lean:62`
- Signature: `theorem sum_ok (s: finset ℚ) : zset.sum (zset.from_set s) = finset.sum s (λ a, a)`
- Description: Shows that `sum` matches its intended semantics.

### `sum_linear`

- Kind: `theorem`
- Source: `src/aggregation.lean:72`
- Signature: `theorem sum_linear (m1 m2: Z[ℚ]) : zset.sum (m1 + m2) = zset.sum m1 + zset.sum m2`
- Description: Shows that `sum` is linear.

### `sum'_ok`

- Kind: `theorem`
- Source: `src/aggregation.lean:81`
- Signature: `theorem sum'_ok (s: finset ℚ) : zset.to_set (sum' (zset.from_set s)) = {finset.sum s (λ a, a)}`
- Description: Shows that `sum'` matches its intended semantics.

## `circuits.lean`

A circuit syntax for stream operators, its denotational semantics, and certified optimization/incrementalization passes over that syntax.

Declarations in this module: **37**.

### `denote`

- Kind: `def`
- Source: `src/circuits.lean:52`
- Signature: `def denote (c: ckt a b) : (stream a → stream b)`
- Description: Interprets a circuit as the stream operator it denotes.

### `equiv`

- Kind: `def`
- Source: `src/circuits.lean:82`
- Signature: `def equiv (f1 f2: ckt a b)`
- Description: Defines circuit equivalence as equality of denotations.

### `equiv_refl`

- Kind: `lemma`
- Source: `src/circuits.lean:87`
- Signature: `lemma equiv_refl (f: ckt a b) : f === f`
- Description: Proves reflexivity of `equiv`.

### `equiv_symm`

- Kind: `lemma`
- Source: `src/circuits.lean:91`
- Signature: `lemma equiv_symm (f1 f2: ckt a b) : f1 === f2 → f2 === f1`
- Description: Proves symmetry of `equiv`.

### `equiv_trans`

- Kind: `lemma`
- Source: `src/circuits.lean:95`
- Signature: `lemma equiv_trans (f1 f2 f3: ckt a b) : f1 === f2 → f2 === f3 → f1 === f3`
- Description: Proves transitivity of `equiv`.

### `denote_seq`

- Kind: `lemma`
- Source: `src/circuits.lean:99`
- Signature: `lemma denote_seq (f1: ckt a b) (f2: ckt b c) : denote (ckt.seq f1 f2) = λ x, denote f2 (denote f1 x)`
- Description: Provides a supporting lemma about `denote_seq`.

### `denote_par`

- Kind: `lemma`
- Source: `src/circuits.lean:103`
- Signature: `lemma denote_par (f1: ckt a b) (f2: ckt c d) : denote (ckt.par f1 f2) = uncurry_op (λ x1 x2, sprod (denote f1 x1, denote f2 x2))`
- Description: Provides a supporting lemma about `denote_par`.

### `denote_delay`

- Kind: `lemma`
- Source: `src/circuits.lean:109`
- Signature: `lemma denote_delay : denote (@ckt.delay a _) = delay`
- Description: Provides a supporting lemma about `denote_delay`.

### `denote_derivative`

- Kind: `lemma`
- Source: `src/circuits.lean:113`
- Signature: `lemma denote_derivative : denote (@ckt.derivative a _) = D`
- Description: Provides a supporting lemma about `denote_derivative`.

### `denote_incremental`

- Kind: `lemma`
- Source: `src/circuits.lean:117`
- Signature: `lemma denote_incremental (f: ckt a b) : denote (ckt.incremental f) = (denote f)^Δ`
- Description: Provides a supporting lemma about `denote_incremental`.

### `denote_integral`

- Kind: `lemma`
- Source: `src/circuits.lean:121`
- Signature: `lemma denote_integral : denote (@ckt.integral a _) = I`
- Description: Provides a supporting lemma about `denote_integral`.

### `denote_lifting`

- Kind: `lemma`
- Source: `src/circuits.lean:125`
- Signature: `lemma denote_lifting (f: Func a b) : denote (ckt.lifting f) = ↑↑(Func_denote f)`
- Description: Provides a supporting lemma about `denote_lifting`.

### `denote_feedback`

- Kind: `lemma`
- Source: `src/circuits.lean:133`
- Signature: `lemma denote_feedback (F: ckt (a × b) b) : denote (ckt.feedback F) = λ s, fix (λ α, denote F (sprod (s, z⁻¹ α)))`
- Description: Provides a supporting lemma about `denote_feedback`.

### `lifting2`

- Kind: `def`
- Source: `src/circuits.lean:151`
- Signature: `def lifting2 (f: a → b → c) : ckt (a × b) c`
- Description: Builds a circuit that lifts a binary pointwise function to streams.

### `derivative`

- Kind: `def`
- Source: `src/circuits.lean:154`
- Signature: `def derivative : ckt a a`
- Description: Defines the derivative circuit.

### `derivative_denote`

- Kind: `theorem`
- Source: `src/circuits.lean:158`
- Signature: `theorem derivative_denote : @derivative a _ === ckt.derivative`
- Description: Provides a supporting lemma about the derivative operator in the stated setting.

### `integral`

- Kind: `def`
- Source: `src/circuits.lean:168`
- Signature: `def integral : ckt a a`
- Description: Defines the integral circuit.

### `integral_denote`

- Kind: `theorem`
- Source: `src/circuits.lean:171`
- Signature: `theorem integral_denote : @integral a _ === ckt.integral`
- Description: Provides a supporting lemma about the integral operator in the stated setting.

### `ckt_causal`

- Kind: `def`
- Source: `src/circuits.lean:182`
- Signature: `def ckt_causal (f: ckt a b) : causal (denote f)`
- Description: Computes a proof that every circuit denotes a causal operator.

### `is_strict`

- Kind: `def`
- Source: `src/circuits.lean:205`
- Signature: `def is_strict (f: ckt a b) : {b:bool | b → strict (denote f)}`
- Description: Computes whether a circuit is strict and, when it is, packages the corresponding proof.

### `incrementalize`

- Kind: `def`
- Source: `src/circuits.lean:244`
- Signature: `def incrementalize (f: ckt a b) : ckt a b`
- Description: Defines a generic recursive optimizer over circuits parameterized by local rewrite choices.

### `incrementalize_ok`

- Kind: `theorem`
- Source: `src/circuits.lean:247`
- Signature: `theorem incrementalize_ok (f: ckt a b) : denote (incrementalize f) = (denote f)^Δ`
- Description: Shows that `incrementalize` matches its intended semantics.

### `seq_assoc`

- Kind: `theorem`
- Source: `src/circuits.lean:255`
- Signature: `theorem seq_assoc (f1: ckt a b) (f2: ckt b c) (f3: ckt c d) : f1 >>> f2 >>> f3 === f1 >>> (f2 >>> f3)`
- Description: Provides a supporting theorem about `seq_assoc`.

### `recursive_opt`

- Kind: `def`
- Source: `src/circuits.lean:267`
- Signature: `def recursive_opt : ckt a b → ckt a b`
- Description: Recursively applies an optional local optimization across an entire circuit.

### `recursive_opt_seq`

- Kind: `lemma`
- Source: `src/circuits.lean:283`
- Signature: `lemma recursive_opt_seq (f1: ckt a b) (f2: ckt b c) : recursive_opt @opt (ckt.seq f1 f2) = (opt $ ckt.seq f1 f2).get_or_else (ckt.seq (recursive_opt @opt f1) (recursive_opt @opt f2))`
- Description: Gives the shape of `recursive_opt` on the named circuit form.

### `recursive_opt_par`

- Kind: `lemma`
- Source: `src/circuits.lean:288`
- Signature: `lemma recursive_opt_par (f1: ckt a b) (f2: ckt c d) : recursive_opt @opt (ckt.par f1 f2) = (opt $ ckt.par f1 f2).get_or_else (ckt.par (recursive_opt @opt f1) (recursive_opt @opt f2))`
- Description: Gives the shape of `recursive_opt` on the named circuit form.

### `recursive_opt_feedback`

- Kind: `lemma`
- Source: `src/circuits.lean:293`
- Signature: `lemma recursive_opt_feedback (f: ckt (a × b) b) : recursive_opt @opt (ckt.feedback f) = (opt $ ckt.feedback f).get_or_else (ckt.feedback (recursive_opt @opt f))`
- Description: Gives the shape of `recursive_opt` on the named circuit form.

### `recursive_opt_incremental`

- Kind: `lemma`
- Source: `src/circuits.lean:298`
- Signature: `lemma recursive_opt_incremental (f: ckt a b) : recursive_opt @opt (ckt.incremental f) = (opt $ ckt.incremental f).get_or_else (ckt.incremental (recursive_opt @opt f))`
- Description: Gives the shape of `recursive_opt` on the named circuit form.

### `opt_or_else_ok`

- Kind: `lemma`
- Source: `src/circuits.lean:309`
- Signature: `lemma opt_or_else_ok (f1 f2: ckt a b) : f2 === f1 → (opt f1).get_or_else f2 === f1`
- Description: Shows that `opt_or_else` matches its intended semantics.

### `recursive_opt_ok`

- Kind: `theorem`
- Source: `src/circuits.lean:319`
- Signature: `theorem recursive_opt_ok : ∀ (f: ckt a b), recursive_opt @opt f === f`
- Description: Shows that `recursive_opt` matches its intended semantics.

### `incrementalize`

- Kind: `def`
- Source: `src/circuits.lean:361`
- Signature: `def incrementalize (c: ckt a b) : ckt a b`
- Description: Builds an optimized incrementalized circuit, preserving linear lifted nodes and incrementalizing the rest.

### `incrementalize_incremental`

- Kind: `lemma`
- Source: `src/circuits.lean:377`
- Signature: `lemma incrementalize_incremental (c: ckt a b) : incrementalize (ckt.incremental c) = ckt.incremental (incrementalize c)`
- Description: Gives the shape of `incrementalize` on the named circuit form.

### `incrementalize_lifting`

- Kind: `lemma`
- Source: `src/circuits.lean:381`
- Signature: `lemma incrementalize_lifting (f: Func a b) : incrementalize (ckt.lifting f) = if is_linear f then ckt.lifting f else ckt.incremental (ckt.lifting f)`
- Description: Gives the shape of `incrementalize` on the named circuit form.

### `incrementalize_seq`

- Kind: `lemma`
- Source: `src/circuits.lean:387`
- Signature: `lemma incrementalize_seq (f1: ckt a b) (f2: ckt b c) : incrementalize (f1 >>> f2) = incrementalize f1 >>> incrementalize f2`
- Description: Gives the shape of `incrementalize` on the named circuit form.

### `incrementalize_par`

- Kind: `lemma`
- Source: `src/circuits.lean:391`
- Signature: `lemma incrementalize_par (f1: ckt a b) (f2: ckt c d) : incrementalize (ckt.par f1 f2) = ckt.par (incrementalize f1) (incrementalize f2)`
- Description: Gives the shape of `incrementalize` on the named circuit form.

### `incrementalize_feedback`

- Kind: `lemma`
- Source: `src/circuits.lean:395`
- Signature: `lemma incrementalize_feedback (f: ckt (a × b) b) : incrementalize (ckt.feedback f) = ckt.feedback (incrementalize f)`
- Description: Gives the shape of `incrementalize` on the named circuit form.

### `incrementalize_ok`

- Kind: `theorem`
- Source: `src/circuits.lean:400`
- Signature: `theorem incrementalize_ok (f: ckt a b) : denote (incrementalize f) = (denote f)^Δ`
- Description: Shows that `incrementalize` matches its intended semantics.

## `incremental.lean`

The core DBSP transformation `Q^Δ = D ∘ Q ∘ I`, together with algebraic laws for composition, feedback, bilinear operators, and nested-stream incrementalization.

Declarations in this module: **65**.

### `incremental`

- Kind: `def`
- Source: `src/incremental.lean:28`
- Signature: `def incremental (Q: operator a b) : operator a b`
- Description: Defines the DBSP incrementalization of a unary stream operator as `D ∘ Q ∘ I`.

### `incremental_unfold`

- Kind: `lemma`
- Source: `src/incremental.lean:32`
- Signature: `lemma incremental_unfold (Q: operator a b) (s: stream a) : incremental Q s = D (Q (I s))`
- Description: Unfolds the definition of `incremental`.

### `incremental2`

- Kind: `def`
- Source: `src/incremental.lean:37`
- Signature: `def incremental2 (T: operator2 a b c) : operator2 a b c`
- Description: Defines the DBSP incrementalization of a curried binary operator.

### `incremental2_unfold`

- Kind: `lemma`
- Source: `src/incremental.lean:41`
- Signature: `lemma incremental2_unfold (Q: operator2 a b c) (s1: stream a) (s2: stream b) : incremental2 Q s1 s2 = D (Q (I s1) (I s2))`
- Description: Unfolds the definition of `incremental2`.

### `incremental_inv`

- Kind: `private def`
- Source: `src/incremental.lean:47`
- Signature: `private def incremental_inv (Q: operator a b) : operator a b`
- Description: Defines the inverse map used to show that incrementalization is a bijection on operators.

### `incremental_inversion_l`

- Kind: `theorem`
- Source: `src/incremental.lean:52`
- Signature: `theorem incremental_inversion_l : function.left_inverse (@incremental a _ b _) incremental_inv`
- Description: Provides a supporting theorem about `incremental_inversion_l`.

### `incremental_inversion_r`

- Kind: `theorem`
- Source: `src/incremental.lean:60`
- Signature: `theorem incremental_inversion_r : function.right_inverse (@incremental a _ b _) incremental_inv`
- Description: Provides a supporting theorem about `incremental_inversion_r`.

### `incremental_bijection`

- Kind: `theorem`
- Source: `src/incremental.lean:68`
- Signature: `theorem incremental_bijection : function.bijective (@incremental a _ b _)`
- Description: Shows that incrementalization is a bijection on unary operators.

### `delay_invariance`

- Kind: `theorem`
- Source: `src/incremental.lean:85`
- Signature: `theorem delay_invariance : incremental (@delay a _) = delay`
- Description: Shows that incrementalizing `delay` leaves it unchanged.

### `integral_invariance`

- Kind: `theorem`
- Source: `src/incremental.lean:92`
- Signature: `theorem integral_invariance : incremental (@I a _) = I`
- Description: Shows that incrementalizing `I` leaves it unchanged.

### `derivative_invariance`

- Kind: `theorem`
- Source: `src/incremental.lean:95`
- Signature: `theorem derivative_invariance : incremental (@D a _) = D`
- Description: Shows that incrementalizing `D` leaves it unchanged.

### `integrate_push`

- Kind: `theorem`
- Source: `src/incremental.lean:98`
- Signature: `theorem integrate_push (Q: operator a b) : Q ∘ I = I ∘ Q^Δ`
- Description: Pushes integration through an operator by rewriting it in incremental form.

### `derivative_push`

- Kind: `theorem`
- Source: `src/incremental.lean:102`
- Signature: `theorem derivative_push (Q: operator a b) : D ∘ Q = Q^Δ ∘ D`
- Description: Pushes differentiation through an operator by rewriting it in incremental form.

### `I_push`

- Kind: `theorem`
- Source: `src/incremental.lean:106`
- Signature: `theorem I_push (Q: operator a b) (s: stream a) : Q (I s) = I (Q^Δ s)`
- Description: Specializes integral push-through to a concrete input stream.

### `D_push`

- Kind: `theorem`
- Source: `src/incremental.lean:110`
- Signature: `theorem D_push (Q: operator a b) (s: stream a) : D (Q s) = Q^Δ (D s)`
- Description: Specializes derivative push-through to a concrete input stream.

### `D_push2`

- Kind: `theorem`
- Source: `src/incremental.lean:114`
- Signature: `theorem D_push2 (Q: operator2 a b c) (s1: stream a) (s2: stream b) : D (Q s1 s2) = Q^Δ2 (D s1) (D s2)`
- Description: Specializes derivative push-through to a concrete binary operator application.

### `chain_incremental`

- Kind: `theorem`
- Source: `src/incremental.lean:118`
- Signature: `theorem chain_incremental (Q1: operator b c) (Q2: operator a b) : (Q1 ∘ Q2)^Δ = Q1^Δ ∘ Q2^Δ`
- Description: Shows that incrementalization distributes over composition.

### `incremental_comp`

- Kind: `theorem`
- Source: `src/incremental.lean:122`
- Signature: `theorem incremental_comp (Q1: operator b c) (Q2: operator a b) (s: stream a) : (λ s, Q1 (Q2 (s)))^Δ s = Q1^Δ (Q2^Δ s)`
- Description: Gives a composition law for `incremental`.

### `add_incremental`

- Kind: `theorem`
- Source: `src/incremental.lean:126`
- Signature: `theorem add_incremental (Q1 Q2: operator a b) : (Q1 + Q2)^Δ = Q1^Δ + Q2^Δ`
- Description: Provides a supporting theorem about `add_incremental`.

### `cycle_body_strict`

- Kind: `lemma`
- Source: `src/incremental.lean:133`
- Signature: `lemma cycle_body_strict (T: operator2 a b b) (hcausal: causal (uncurry_op T)) : ∀ s, strict (λ (α : stream b), T s (z⁻¹ α))`
- Description: Shows that `cycle_body` is strict.

### `cycle_body_integral_strict`

- Kind: `lemma`
- Source: `src/incremental.lean:141`
- Signature: `lemma cycle_body_integral_strict (T: operator2 a b b) (hcausal: causal (uncurry_op T)) : ∀ s, strict (λ (α : stream b), T (I s) (z⁻¹ α))`
- Description: Shows that `cycle_body_integral` is strict.

### `cycle_body_incremental_strict`

- Kind: `lemma`
- Source: `src/incremental.lean:149`
- Signature: `lemma cycle_body_incremental_strict (T: operator2 a b b) (hcausal: causal (uncurry_op T)) (s: stream a) : strict (λ α, T^Δ2 s (z⁻¹ α))`
- Description: Shows that `cycle_body_incremental` is strict.

### `cycle_incremental`

- Kind: `theorem`
- Source: `src/incremental.lean:159`
- Signature: `theorem cycle_incremental (T: operator2 a b b) (hcausal: causal (uncurry_op T)) : (λ (s: stream a), fix (λ α, T s (z⁻¹ α)))^Δ = λ s, fix (λ α, T^Δ2 s (z⁻¹ α))`
- Description: Shows that incrementalization commutes with the one-dimensional feedback construction.

### `incremental_sprod`

- Kind: `lemma`
- Source: `src/incremental.lean:174`
- Signature: `lemma incremental_sprod (f: operator (a×b) c) (s1: stream a) (s2: stream b) : f^Δ (sprod (s1, s2)) = (λ s1 s2, f (sprod (s1, s2)))^Δ2 s1 s2`
- Description: Provides a supporting lemma about `incremental_sprod`.

### `lifting_cycle_body_strict2`

- Kind: `theorem`
- Source: `src/incremental.lean:181`
- Signature: `theorem lifting_cycle_body_strict2 (T: operator2 a b b) (hcausal: causal (uncurry_op T)) : ∀ s, strict2 (λ (α : stream (stream b)), ↑²T s (↑↑z⁻¹ α))`
- Description: Provides a supporting lemma about lifted operators.

### `sum_vals_nested`

- Kind: `lemma`
- Source: `src/incremental.lean:196`
- Signature: `lemma sum_vals_nested (s: stream (stream a)) (n t: ℕ) : sum_vals s n t = sum_vals (λ n, s n t) n`
- Description: Lifts the corresponding property to the nested-stream setting for `sum_vals`.

### `integral_lift_time_invariant`

- Kind: `lemma`
- Source: `src/incremental.lean:202`
- Signature: `lemma integral_lift_time_invariant (s: stream (stream a)) : I (↑↑z⁻¹ s) = ↑↑z⁻¹ (I s)`
- Description: Shows that `integral_lift` is time-invariant.

### `lift_integral_lift_time_invariant`

- Kind: `lemma`
- Source: `src/incremental.lean:219`
- Signature: `lemma lift_integral_lift_time_invariant (s: stream (stream a)) : ↑↑I (↑↑z⁻¹ s) = ↑↑z⁻¹ (↑↑I s)`
- Description: Shows that `lift_integral_lift` is time-invariant.

### `lifting_delay_linear`

- Kind: `lemma`
- Source: `src/incremental.lean:226`
- Signature: `lemma lifting_delay_linear : linear (↑↑(@delay a _))`
- Description: Shows that `lifting_delay` is linear.

### `integral_causal_nested'`

- Kind: `lemma`
- Source: `src/incremental.lean:232`
- Signature: `lemma integral_causal_nested' (s1 s2: stream (stream a)) (n t: ℕ) (heq: ∀ n' ≤ n, s1 n' t = s2 n' t) : I s1 n t = I s2 n t`
- Description: Provides a supporting lemma about the integral operator in the stated setting.

### `integral_causal_nested`

- Kind: `lemma`
- Source: `src/incremental.lean:246`
- Signature: `lemma integral_causal_nested : causal_nested (@I (stream a) _)`
- Description: Lifts the corresponding property to the nested-stream setting for `integral_causal`.

### `derivative_causal_nested`

- Kind: `lemma`
- Source: `src/incremental.lean:254`
- Signature: `lemma derivative_causal_nested : causal_nested (@D (stream a) _)`
- Description: Lifts the corresponding property to the nested-stream setting for `derivative_causal`.

### `cycle_body_strict2`

- Kind: `lemma`
- Source: `src/incremental.lean:264`
- Signature: `lemma cycle_body_strict2 (T: operator2 a (stream b) (stream b)) (s: stream a) : causal_nested (T s) → strict2 (λ α, T s (↑↑z⁻¹ α))`
- Description: Provides a supporting lemma about `cycle_body_strict2`.

### `cycle_body_incremental_strict2`

- Kind: `lemma`
- Source: `src/incremental.lean:277`
- Signature: `lemma cycle_body_incremental_strict2 (T: operator2 a (stream b) (stream b)) (s: stream a) : causal_nested (T (I s)) → strict2 (λ α, T^Δ2 s (↑↑z⁻¹ α))`
- Description: Provides a supporting lemma about `cycle_body_incremental_strict2`.

### `lifting_cycle`

- Kind: `theorem`
- Source: `src/incremental.lean:293`
- Signature: `theorem lifting_cycle (T: operator2 a b b) (hcausal: causal (uncurry_op T)) : ↑↑(λ s, fix (λ α, T s (z⁻¹ α))) = λ s, fix2 (λ α, ↑²T s (↑↑z⁻¹ α))`
- Description: Shows that lifting incrementalization through a nested feedback construction is sound.

### `cycle2_incremental`

- Kind: `theorem`
- Source: `src/incremental.lean:304`
- Signature: `theorem cycle2_incremental (T: operator2 a (stream b) (stream b)) (hcausal: ∀ s, causal_nested (T s)) : (λ (s: stream a), fix2 (λ α, T s (↑↑z⁻¹ α)))^Δ = λ s, fix2 (λ α, T^Δ2 s (↑↑z⁻¹ α))`
- Description: Shows that incrementalization commutes with the nested-stream feedback construction.

### `lti_incremental`

- Kind: `theorem`
- Source: `src/incremental.lean:320`
- Signature: `theorem lti_incremental (Q: operator a b) (h: lti Q) : Q^Δ = Q`
- Description: Shows that linear time-invariant operators are fixed points of incrementalization.

### `I_incremental`

- Kind: `theorem`
- Source: `src/incremental.lean:333`
- Signature: `theorem I_incremental : I^Δ = @I a _`
- Description: Characterizes the incremental form of integration.

### `D_incremental`

- Kind: `theorem`
- Source: `src/incremental.lean:341`
- Signature: `theorem D_incremental : D^Δ = @D a _`
- Description: Characterizes the incremental form of differentiation.

### `delay_lti`

- Kind: `theorem`
- Source: `src/incremental.lean:348`
- Signature: `theorem delay_lti : lti (@delay a _)`
- Description: Shows that `delay` is linear and time-invariant.

### `delay_incremental`

- Kind: `theorem`
- Source: `src/incremental.lean:357`
- Signature: `theorem delay_incremental : z⁻¹^Δ = @delay a _`
- Description: Characterizes the incremental form of delay.

### `sprod_time_invariant`

- Kind: `lemma`
- Source: `src/incremental.lean:364`
- Signature: `lemma sprod_time_invariant (T : operator2 a b c) (s1 : stream a) (s2 : stream b) : ↑(z⁻¹ s1, z⁻¹ s2) = z⁻¹ (↑(s1, s2) : stream (a × b))`
- Description: Shows that `sprod` is time-invariant.

### `time_invariant_map_fst`

- Kind: `lemma`
- Source: `src/incremental.lean:373`
- Signature: `lemma time_invariant_map_fst (s: stream (a × b)) (n: ℕ) : (z⁻¹ s n).fst = z⁻¹ (λ (n : ℕ), (s n).fst) n`
- Description: Provides a supporting lemma about `time_invariant_map_fst`.

### `time_invariant_map_snd`

- Kind: `lemma`
- Source: `src/incremental.lean:380`
- Signature: `lemma time_invariant_map_snd (s: stream (a × b)) (n: ℕ) : (z⁻¹ s n).snd = z⁻¹ (λ (n : ℕ), (s n).snd) n`
- Description: Provides a supporting lemma about `time_invariant_map_snd`.

### `time_invariant2`

- Kind: `lemma`
- Source: `src/incremental.lean:387`
- Signature: `lemma time_invariant2 (T: operator2 a b c) : time_invariant (uncurry_op T) ↔ (∀ s1 s2, T (z⁻¹ s1) (z⁻¹ s2) = z⁻¹ (T s1 s2))`
- Description: Provides a supporting lemma about `time_invariant2`.

### `causal_incremental`

- Kind: `theorem`
- Source: `src/incremental.lean:408`
- Signature: `theorem causal_incremental (Q: operator a b) : causal Q → causal (Q^Δ)`
- Description: Shows that incrementalization preserves causality.

### `causal_incremental2`

- Kind: `theorem`
- Source: `src/incremental.lean:419`
- Signature: `theorem causal_incremental2 (Q: operator2 a b c) : causal (uncurry_op Q) → causal (λ s, Q^Δ2 (↑↑prod.fst s) (↑↑prod.snd s))`
- Description: Shows that binary incrementalization preserves causality.

### `causal_nested_incremental`

- Kind: `theorem`
- Source: `src/incremental.lean:439`
- Signature: `theorem causal_nested_incremental (Q: operator (stream a) (stream b)) : causal_nested Q → causal_nested (Q^Δ)`
- Description: Shows that nested incrementalization preserves nested causality.

### `causal_nested_lifting2`

- Kind: `theorem`
- Source: `src/incremental.lean:454`
- Signature: `theorem causal_nested_lifting2 {d: Type} [add_comm_group d] (f: stream b → stream c → stream d) (g: operator (stream a) (stream b)) (h: operator (stream a) (stream c)) : causal (uncurry_op f) → causal_nested g → causal_nested h → causal_nested (λ s, ↑²f (g s) (h s))`
- Description: Provides a supporting lemma about causality in the stated setting.

### `causal_nested_lifting2_incremental`

- Kind: `theorem`
- Source: `src/incremental.lean:470`
- Signature: `theorem causal_nested_lifting2_incremental {d: Type} [add_comm_group d] (f: stream b → stream c → stream d) (g: operator (stream a) (stream b)) (h: operator (stream a) (stream c)) : causal (uncurry_op f) → causal_nested g → causal_nested h → causal_nested (λ s, ↑²f^Δ2 (g s) (h s))`
- Description: Provides a supporting lemma about causality in the stated setting.

### `causal_lifting2`

- Kind: `theorem`
- Source: `src/incremental.lean:487`
- Signature: `theorem causal_lifting2 {d: Type} [add_comm_group d] (f: b → c → d) (g: operator a b) (h: operator a c) : causal g → causal h → causal (λ s, ↑²f (g s) (h s))`
- Description: Provides a supporting lemma about causality in the stated setting.

### `causal_lifting2_incremental`

- Kind: `theorem`
- Source: `src/incremental.lean:499`
- Signature: `theorem causal_lifting2_incremental {d: Type} [add_comm_group d] (f: b → c → d) (g: operator a b) (h: operator a c) : causal g → causal h → causal (λ s, ↑²f^Δ2 (g s) (h s))`
- Description: Provides a supporting lemma about causality in the stated setting.

### `lifting2_sum`

- Kind: `lemma`
- Source: `src/incremental.lean:512`
- Signature: `lemma lifting2_sum (f g: a → b → c) : (↑² (λ x y, f x y + g x y)) = ↑²f + ↑²g`
- Description: Provides a supporting lemma about doubly lifted operators and their incremental forms.

### `lifting2_incremental_sum`

- Kind: `theorem`
- Source: `src/incremental.lean:515`
- Signature: `theorem lifting2_incremental_sum (f g: a → b → c) : (↑² (λ x y, f x y + g x y))^Δ2 = ↑²f^Δ2 + ↑²g^Δ2`
- Description: Provides a supporting lemma about doubly lifted operators and their incremental forms.

### `lifting2_incremental_unfold`

- Kind: `private lemma`
- Source: `src/incremental.lean:524`
- Signature: `private lemma lifting2_incremental_unfold {d e: Type} [add_comm_group d] [add_comm_group e] (f: b → c → d) (g: a → b) (h: e → c) : ↑²(λ x y, f (g x) (h y))^Δ2 = λ s1 s2, D (↑²f (↑↑g (I s1)) (↑↑h (I s2)))`
- Description: Internal helper: unfolds the definition of `lifting2_incremental`.

### `lifting2_incremental_comp`

- Kind: `lemma`
- Source: `src/incremental.lean:531`
- Signature: `lemma lifting2_incremental_comp {d e: Type} [add_comm_group d] [add_comm_group e] (f: b → c → d) (g: a → b) (h: e → c) : ↑²(λ x y, f (g x) (h y))^Δ2 = λ s1 s2, ↑²f^Δ2 (↑↑g^Δ s1) (↑↑h^Δ s2)`
- Description: Gives a composition law for `lifting2_incremental`.

### `incremental_id`

- Kind: `lemma`
- Source: `src/incremental.lean:543`
- Signature: `lemma incremental_id : incremental (λ (x:stream a), x) = id`
- Description: Provides a supporting lemma about `incremental_id`.

### `incremental_id'`

- Kind: `lemma`
- Source: `src/incremental.lean:548`
- Signature: `lemma incremental_id' : incremental (@id (stream a)) = id`
- Description: Provides a supporting lemma about `incremental_id'`.

### `lifting2_incremental_comp_1'`

- Kind: `lemma`
- Source: `src/incremental.lean:552`
- Signature: `lemma lifting2_incremental_comp_1' {d: Type} [add_comm_group d] (f: b → c → d) (g: a → b) : ↑²(λ x, f (g x))^Δ2 = λ s1, ↑²f^Δ2 (↑↑g^Δ s1)`
- Description: Provides a supporting lemma about doubly lifted operators and their incremental forms.

### `lifting2_incremental_comp_1`

- Kind: `lemma`
- Source: `src/incremental.lean:560`
- Signature: `lemma lifting2_incremental_comp_1 {d: Type} [add_comm_group d] (f: b → c → d) (g: a → b) : ↑²(λ x y, f (g x) y)^Δ2 = λ s1 s2, ↑²f^Δ2 (↑↑g^Δ s1) s2`
- Description: Provides a supporting lemma about doubly lifted operators and their incremental forms.

### `lifting2_incremental_comp_2`

- Kind: `lemma`
- Source: `src/incremental.lean:568`
- Signature: `lemma lifting2_incremental_comp_2 {d e: Type} [add_comm_group d] [add_comm_group e] (f: b → c → d) (h: e → c) : ↑²(λ x y, f x (h y))^Δ2 = λ s1 s2, ↑²f^Δ2 s1 (↑↑h^Δ s2)`
- Description: Provides a supporting lemma about doubly lifted operators and their incremental forms.

### `times_incremental`

- Kind: `def`
- Source: `src/incremental.lean:594`
- Signature: `def times_incremental : stream α → stream β → stream γ`
- Description: Defines the standard incremental formula for a bilinear binary operator.

### `bilinear_incremental`

- Kind: `theorem`
- Source: `src/incremental.lean:597`
- Signature: `theorem bilinear_incremental : time_invariant (uncurry_op times) → bilinear times → times^Δ2 = times_incremental times`
- Description: Derives the standard DBSP incremental rule for bilinear operators.

### `bilinear_incremental_forward_proof`

- Kind: `theorem`
- Source: `src/incremental.lean:632`
- Signature: `theorem bilinear_incremental_forward_proof : time_invariant (uncurry_op times) → bilinear times → ∀ a b, times^Δ2 a b = a ** b + I (z⁻¹ a) ** b + a ** I (z⁻¹ b)`
- Description: Gives a forward equational proof of the bilinear incremental rule.

### `bilinear_incremental_short_paper_proof`

- Kind: `theorem`
- Source: `src/incremental.lean:656`
- Signature: `theorem bilinear_incremental_short_paper_proof : time_invariant (uncurry_op times) → bilinear times → ∀ a b, times^Δ2 a b = a ** b + I (z⁻¹ a) ** b + a ** I (z⁻¹ b)`
- Description: Restates the bilinear incremental proof in the compact style used in the paper.

## `linear.lean`

Linearity, bilinearity, derivative/integral operators, and the feedback operator over additive streams.

Declarations in this module: **67**.

### `linear`

- Kind: `def`
- Source: `src/linear.lean:26`
- Signature: `def linear (S: operator a b)`
- Description: Defines linearity for stream operators over additive groups.

### `linear_add`

- Kind: `lemma`
- Source: `src/linear.lean:32`
- Signature: `lemma linear_add {S: operator a b} (h: linear S) : ∀ s1 s2, S (s1 + s2) = S s1 + S s2`
- Description: Provides a supporting lemma about `linear_add`.

### `linear_zero`

- Kind: `lemma`
- Source: `src/linear.lean:35`
- Signature: `lemma linear_zero {S: operator a b} (h: linear S) : S 0 = 0`
- Description: Shows the zero-case behavior of `linear`.

### `linear_neg`

- Kind: `lemma`
- Source: `src/linear.lean:43`
- Signature: `lemma linear_neg {S: operator a b} (h: linear S) : ∀ s, S (-s) = -S s`
- Description: Provides a supporting lemma about `linear_neg`.

### `linear_sub`

- Kind: `lemma`
- Source: `src/linear.lean:52`
- Signature: `lemma linear_sub {S: operator a b} (h: linear S) : ∀ s1 s2, S (s1 - s2) = S s1 - S s2`
- Description: Provides a supporting lemma about `linear_sub`.

### `lifted_linear`

- Kind: `theorem`
- Source: `src/linear.lean:61`
- Signature: `theorem lifted_linear (f: a → b) : (∀ x y, f (x + y) = f x + f y) → linear (lifting f)`
- Description: Shows that `lifted` is linear.

### `lti`

- Kind: `def`
- Source: `src/linear.lean:70`
- Signature: `def lti (S: operator a b)`
- Description: Defines the combined property of linearity plus time invariance.

### `lti_operator_zpp`

- Kind: `lemma`
- Source: `src/linear.lean:74`
- Signature: `lemma lti_operator_zpp (S: operator a b) : lti S → S 0 0 = 0`
- Description: Establishes the zero-at-zero property for `lti_operator`.

### `bilinear`

- Kind: `def`
- Source: `src/linear.lean:79`
- Signature: `def bilinear (f: a → b → c)`
- Description: Defines bilinearity for binary stream operators.

### `bilinear_sub_1`

- Kind: `lemma`
- Source: `src/linear.lean:84`
- Signature: `lemma bilinear_sub_1 {f: operator2 a b c} (hblin: bilinear f) : ∀ x1 x2 y, f (x1 - x2) y = f x1 y - f x2 y`
- Description: Provides a supporting lemma about `bilinear_sub_1`.

### `bilinear_sub_2`

- Kind: `lemma`
- Source: `src/linear.lean:96`
- Signature: `lemma bilinear_sub_2 {f: operator2 a b c} (hblin: bilinear f) : ∀ x y1 y2, f x (y1 - y2) = f x y1 - f x y2`
- Description: Provides a supporting lemma about `bilinear_sub_2`.

### `lifting_bilinear`

- Kind: `theorem`
- Source: `src/linear.lean:104`
- Signature: `theorem lifting_bilinear (f: a → b → c) : bilinear f → bilinear ↑²f`
- Description: Shows that `lifting` is bilinear.

### `mul_Z_bilinear`

- Kind: `theorem`
- Source: `src/linear.lean:118`
- Signature: `theorem mul_Z_bilinear : bilinear (lifting2 (λ (z1 z2 : ℤ), z1 * z2))`
- Description: Shows that `mul_Z` is bilinear.

### `mul_ring_bilinear`

- Kind: `theorem`
- Source: `src/linear.lean:128`
- Signature: `theorem mul_ring_bilinear {a: Type} [ring a] : bilinear (@lifting2 a a a (λ (z1 z2 : a), z1 * z2))`
- Description: Shows that `mul_ring` is bilinear.

### `feedback`

- Kind: `def`
- Source: `src/linear.lean:152`
- Signature: `def feedback (S: operator a a) : operator a a`
- Description: Defines the feedback operator used to close a recursive circuit around a delayed state.

### `feedback_strict`

- Kind: `theorem`
- Source: `src/linear.lean:155`
- Signature: `theorem feedback_strict {S : operator a a} (hcausal : causal S) (s : stream a) : strict (λ (α : stream a), S (s + delay α))`
- Description: Shows strictness of the feedback body needed for fixpoint reasoning.

### `feedback_unfold`

- Kind: `theorem`
- Source: `src/linear.lean:169`
- Signature: `theorem feedback_unfold (S: operator a a) : causal S → ∀ s, feedback S s = S (s + delay (feedback S s))`
- Description: Unfolds the definition of `feedback`.

### `delay_linear`

- Kind: `lemma`
- Source: `src/linear.lean:179`
- Signature: `lemma delay_linear : linear (@delay a _)`
- Description: Shows that `delay` is linear.

### `add_linear`

- Kind: `theorem`
- Source: `src/linear.lean:189`
- Signature: `theorem add_linear : linear (uncurry_op ((+) : stream a → stream a → stream a))`
- Description: Shows that `add` is linear.

### `agree_upto_respects_add`

- Kind: `lemma`
- Source: `src/linear.lean:198`
- Signature: `lemma agree_upto_respects_add (s1 s2 s1' s2': stream a) (n: ℕ) : s1 ==n== s1' → s2 ==n== s2' → (s1 + s2) ==n== (s1' + s2')`
- Description: Provides a supporting lemma about finite-horizon stream agreement.

### `agree_upto_respects_sub`

- Kind: `lemma`
- Source: `src/linear.lean:208`
- Signature: `lemma agree_upto_respects_sub (s1 s2 s1' s2': stream a) (n: ℕ) : s1 ==n== s1' → s2 ==n== s2' → (s1 - s2) ==n== (s1' - s2')`
- Description: Provides a supporting lemma about finite-horizon stream agreement.

### `feedback_time_invariant`

- Kind: `theorem`
- Source: `src/linear.lean:220`
- Signature: `theorem feedback_time_invariant (S: operator a a) : causal S → time_invariant S → time_invariant (feedback S)`
- Description: Shows that feedback preserves time invariance.

### `feedback_causal`

- Kind: `theorem`
- Source: `src/linear.lean:247`
- Signature: `theorem feedback_causal (S: operator a a) : causal S → causal (feedback S)`
- Description: Shows that feedback preserves causality.

### `feedback_linear`

- Kind: `theorem`
- Source: `src/linear.lean:270`
- Signature: `theorem feedback_linear (S: operator a a) : causal S → lti S → linear (feedback S)`
- Description: Shows that feedback preserves linearity under the stated strictness assumptions.

### `feedback_lti`

- Kind: `theorem`
- Source: `src/linear.lean:287`
- Signature: `theorem feedback_lti (S: operator a a) : causal S → lti S → lti (feedback S)`
- Description: Shows that feedback preserves linear time invariance.

### `D`

- Kind: `def`
- Source: `src/linear.lean:304`
- Signature: `def D : operator a a`
- Description: Defines the stream derivative operator.

### `derivative_causal`

- Kind: `lemma`
- Source: `src/linear.lean:307`
- Signature: `lemma derivative_causal : causal (@D a _)`
- Description: Shows that `derivative` is causal.

### `derivative_time_invariant`

- Kind: `lemma`
- Source: `src/linear.lean:318`
- Signature: `lemma derivative_time_invariant : time_invariant (@D a _)`
- Description: Shows that `derivative` is time-invariant.

### `derivative_linear`

- Kind: `lemma`
- Source: `src/linear.lean:325`
- Signature: `lemma derivative_linear : linear (@D a _)`
- Description: Shows that `derivative` is linear.

### `derivative_lti`

- Kind: `lemma`
- Source: `src/linear.lean:335`
- Signature: `lemma derivative_lti : lti (@D a _)`
- Description: Shows that `derivative` is linear and time-invariant.

### `I`

- Kind: `def`
- Source: `src/linear.lean:346`
- Signature: `def I : operator a a`
- Description: Defines the stream integral operator.

### `id_causal`

- Kind: `protected lemma`
- Source: `src/linear.lean:348`
- Signature: `protected lemma id_causal : causal (@id (stream a))`
- Description: Shows that `id` is causal.

### `id_time_invariant`

- Kind: `protected lemma`
- Source: `src/linear.lean:355`
- Signature: `protected lemma id_time_invariant : time_invariant (@id (stream a))`
- Description: Shows that `id` is time-invariant.

### `id_lti`

- Kind: `protected lemma`
- Source: `src/linear.lean:361`
- Signature: `protected lemma id_lti : lti (@id (stream a))`
- Description: Shows that `id` is linear and time-invariant.

### `integral_causal`

- Kind: `lemma`
- Source: `src/linear.lean:369`
- Signature: `lemma integral_causal : causal (@I a _)`
- Description: Shows that `integral` is causal.

### `integral_lti`

- Kind: `theorem`
- Source: `src/linear.lean:375`
- Signature: `theorem integral_lti : lti (@I a _)`
- Description: Shows that `integral` is linear and time-invariant.

### `integral_time_invariant`

- Kind: `theorem`
- Source: `src/linear.lean:383`
- Signature: `theorem integral_time_invariant : time_invariant (@I a _)`
- Description: Shows that `integral` is time-invariant.

### `integral_linear`

- Kind: `theorem`
- Source: `src/linear.lean:386`
- Signature: `theorem integral_linear : linear (@I a _)`
- Description: Shows that `integral` is linear.

### `integral_unfold`

- Kind: `theorem`
- Source: `src/linear.lean:389`
- Signature: `theorem integral_unfold : ∀ (s: stream a), I s = s + delay (I s)`
- Description: Unfolds the definition of `integral`.

### `sum_vals`

- Kind: `def`
- Source: `src/linear.lean:400`
- Signature: `def sum_vals (s: stream a) : ℕ → a | 0`
- Description: Defines prefix summation over stream values.

### `sum_vals_0`

- Kind: `lemma`
- Source: `src/linear.lean:405`
- Signature: `lemma sum_vals_0 (s: stream a) : sum_vals s 0 = 0`
- Description: Shows the zero-case behavior of `sum_vals`.

### `sum_vals_1`

- Kind: `lemma`
- Source: `src/linear.lean:408`
- Signature: `lemma sum_vals_1 (s: stream a) : sum_vals s 1 = s 0`
- Description: Provides a supporting summation lemma about `sum_vals_1`.

### `sum_vals_zero`

- Kind: `lemma`
- Source: `src/linear.lean:416`
- Signature: `lemma sum_vals_zero (s: stream a) : (∀ n, s n = 0) → ∀ (n:ℕ), sum_vals s n = 0`
- Description: Shows the zero-case behavior of `sum_vals`.

### `integral_0`

- Kind: `lemma`
- Source: `src/linear.lean:427`
- Signature: `lemma integral_0 (s: stream a) : I s 0 = s 0`
- Description: Shows the zero-case behavior of `integral`.

### `integral_sum_vals`

- Kind: `theorem`
- Source: `src/linear.lean:432`
- Signature: `theorem integral_sum_vals (s: stream a) (n: ℕ) : I s n = sum_vals s n.succ`
- Description: Provides a supporting lemma about the integral operator in the stated setting.

### `integral_zpp`

- Kind: `lemma`
- Source: `src/linear.lean:442`
- Signature: `lemma integral_zpp : I (0: stream a) = 0`
- Description: Establishes the zero-at-zero property for `integral`.

### `derivative_0`

- Kind: `lemma`
- Source: `src/linear.lean:451`
- Signature: `lemma derivative_0 (s: stream a) : D s 0 = s 0`
- Description: Shows the zero-case behavior of `derivative`.

### `derivative_difference_t`

- Kind: `theorem`
- Source: `src/linear.lean:457`
- Signature: `theorem derivative_difference_t (s: stream a) (t: ℕ) : 0 < t → D s t = s t - s (t - 1)`
- Description: Provides a supporting lemma about the derivative operator in the stated setting.

### `derivative_zpp`

- Kind: `lemma`
- Source: `src/linear.lean:467`
- Signature: `lemma derivative_zpp : D (0: stream a) = 0`
- Description: Establishes the zero-at-zero property for `derivative`.

### `add_causal`

- Kind: `lemma`
- Source: `src/linear.lean:472`
- Signature: `lemma add_causal : causal (uncurry_op ((+) : operator2 a a a))`
- Description: Shows that `add` is causal.

### `sum_causal`

- Kind: `lemma`
- Source: `src/linear.lean:479`
- Signature: `lemma sum_causal (f g: operator a b) : causal f → causal g → causal (λ x, f x + g x)`
- Description: Shows that `sum` is causal.

### `sum_causal_nested`

- Kind: `lemma`
- Source: `src/linear.lean:488`
- Signature: `lemma sum_causal_nested (f g: operator (stream a) (stream b)) : causal_nested f → causal_nested g → causal_nested (λ x, f x + g x)`
- Description: Lifts the corresponding property to the nested-stream setting for `sum_causal`.

### `sum_vals_succ_n`

- Kind: `lemma`
- Source: `src/linear.lean:497`
- Signature: `lemma sum_vals_succ_n (s: stream a) (t: ℕ) : sum_vals (D s) t.succ = s t`
- Description: Provides a supporting summation lemma about `sum_vals_succ_n`.

### `derivative_integral`

- Kind: `theorem`
- Source: `src/linear.lean:508`
- Signature: `theorem derivative_integral (s: stream a) : I (D s) = s`
- Description: Shows that differentiating an integral recovers the original stream.

### `derivative_integral_alt`

- Kind: `private theorem`
- Source: `src/linear.lean:518`
- Signature: `private theorem derivative_integral_alt (s: stream a) : I (D s) = s`
- Description: Internal helper: provides a supporting lemma about the derivative operator in the stated setting.

### `derivative_integral_alt2`

- Kind: `private theorem`
- Source: `src/linear.lean:532`
- Signature: `private theorem derivative_integral_alt2 (s: stream a) : I (D s) = s`
- Description: Internal helper: provides a supporting lemma about the derivative operator in the stated setting.

### `integral_derivative`

- Kind: `theorem`
- Source: `src/linear.lean:542`
- Signature: `theorem integral_derivative (s: stream a) : D (I s) = s`
- Description: Shows that integrating a derivative recovers the original stream under the stated conditions.

### `derivative_integral_inverse`

- Kind: `theorem`
- Source: `src/linear.lean:551`
- Signature: `theorem derivative_integral_inverse (α s: stream a): α = I s ↔ D α = s`
- Description: Packages the derivative/integral relationship as an inverse law.

### `i_d_comp`

- Kind: `lemma`
- Source: `src/linear.lean:562`
- Signature: `lemma i_d_comp : I ∘ D = @id (stream a)`
- Description: States the composition law for `I ∘ D`.

### `d_i_comp`

- Kind: `lemma`
- Source: `src/linear.lean:568`
- Signature: `lemma d_i_comp : D ∘ I = @id (stream a)`
- Description: States the composition law for `D ∘ I`.

### `lifting_linear`

- Kind: `theorem`
- Source: `src/linear.lean:574`
- Signature: `theorem lifting_linear (f: a → b) : (∀ x y, f (x + y) = f x + f y) → linear (↑↑f)`
- Description: Shows that `lifting` is linear.

### `lifting_lti`

- Kind: `theorem`
- Source: `src/linear.lean:583`
- Signature: `theorem lifting_lti (f: a → b) : (∀ x y, f (x + y) = f x + f y) → lti (↑↑ f)`
- Description: Shows that `lifting` is linear and time-invariant.

### `derivative_sprod`

- Kind: `lemma`
- Source: `src/linear.lean:595`
- Signature: `lemma derivative_sprod (s1: stream a) (s2: stream b) : D (sprod (s1, s2)) = sprod (D s1, D s2)`
- Description: Provides a supporting lemma about the derivative operator in the stated setting.

### `integral_sprod`

- Kind: `lemma`
- Source: `src/linear.lean:606`
- Signature: `lemma integral_sprod (s1: stream a) (s2: stream b) : I (sprod (s1, s2)) = sprod (I s1, I s2)`
- Description: Provides a supporting lemma about the integral operator in the stated setting.

### `integral_lift_comm`

- Kind: `lemma`
- Source: `src/linear.lean:617`
- Signature: `lemma integral_lift_comm (f: a → b) (s: stream a) : (∀ x y, f (x + y) = f x + f y) → I (↑↑f s) = ↑↑f (I s)`
- Description: Shows that `integral_lift` commutes with the compared operation.

### `integral_fst_comm`

- Kind: `lemma`
- Source: `src/linear.lean:631`
- Signature: `lemma integral_fst_comm (s: stream (a × b)) : I (↑↑prod.fst s) = ↑↑prod.fst (I s)`
- Description: Shows that `integral_fst` commutes with the compared operation.

### `integral_snd_comm`

- Kind: `lemma`
- Source: `src/linear.lean:638`
- Signature: `lemma integral_snd_comm (s: stream (a × b)) : I (↑↑prod.snd s) = ↑↑prod.snd (I s)`
- Description: Shows that `integral_snd` commutes with the compared operation.

## `operators.lean`

Core stream-operator vocabulary: lifting, delay, time invariance, causality, strictness, and fixpoint constructions over streams and nested streams.

Declarations in this module: **91**.

### `operator`

- Kind: `def`
- Source: `src/operators.lean:29`
- Signature: `def operator (a b: Type u) : Type u`
- Description: Defines a unary operator as a function from input streams to output streams.

### `operator2`

- Kind: `def`
- Source: `src/operators.lean:36`
- Signature: `def operator2 (a b c: Type u) : Type u`
- Description: Defines a curried binary operator on streams.

### `lifting`

- Kind: `def`
- Source: `src/operators.lean:43`
- Signature: `def lifting {a b: Type} (f: a → b) : operator a b`
- Description: Lifts a pointwise function to an operator on streams.

### `lifting_eq`

- Kind: `lemma`
- Source: `src/operators.lean:49`
- Signature: `lemma lifting_eq {a b: Type} (f: a → b) (s: stream a) (n: ℕ) : ↑↑f s n = f (s n)`
- Description: States an equality characterizing `lifting`.

### `lifting2`

- Kind: `def`
- Source: `src/operators.lean:53`
- Signature: `def lifting2 {a b c: Type} (f: a → b → c) : operator2 a b c`
- Description: Lifts a binary pointwise function to a binary operator on streams.

### `lifting2_apply`

- Kind: `lemma`
- Source: `src/operators.lean:57`
- Signature: `lemma lifting2_apply {a b c: Type} (f: a → b → c) (s1: stream a) (s2: stream b) (n: ℕ) : lifting2 f s1 s2 n = f (s1 n) (s2 n)`
- Description: Gives the pointwise evaluation rule for `lifting2`.

### `lifting_id`

- Kind: `lemma`
- Source: `src/operators.lean:62`
- Signature: `lemma lifting_id {a: Type} : lifting (λ (x:a), x) = id`
- Description: Provides a supporting lemma about lifted operators.

### `sprod`

- Kind: `def`
- Source: `src/operators.lean:89`
- Signature: `def sprod {a b: Type} : stream a × stream b → stream (a × b)`
- Description: Combines two streams into a stream of pairs by zipping them pointwise.

### `sprod_apply`

- Kind: `lemma`
- Source: `src/operators.lean:97`
- Signature: `lemma sprod_apply (s: stream a × stream b) (n: ℕ) : (sprod s) n = (s.1 n, s.2 n)`
- Description: Gives the pointwise evaluation rule for `sprod`.

### `sprod_coe_unfold`

- Kind: `lemma`
- Source: `src/operators.lean:101`
- Signature: `lemma sprod_coe_unfold (s: stream a × stream b) (n: ℕ) : (↑s : stream (a × b)) n = (s.1 n, s.2 n)`
- Description: Unfolds the definition of `sprod_coe`.

### `uncurry_op`

- Kind: `def`
- Source: `src/operators.lean:109`
- Signature: `def uncurry_op (T: operator2 a b c) : operator (a × b) c`
- Description: Turns a curried binary operator into an operator over paired streams.

### `uncurry_op_intro`

- Kind: `lemma`
- Source: `src/operators.lean:112`
- Signature: `lemma uncurry_op_intro (T: operator2 a b c) (s1: stream a) (s2: stream b) : T s1 s2 = uncurry_op T (s1, s2)`
- Description: Gives an introduction rule for `uncurry_op`.

### `lifting_distributivity`

- Kind: `theorem`
- Source: `src/operators.lean:115`
- Signature: `theorem lifting_distributivity {a b c: Type} (f: a → b) (g: b → c) : lifting (g ∘ f) = lifting g ∘ lifting f`
- Description: Provides a supporting lemma about lifted operators.

### `lifting_comp`

- Kind: `theorem`
- Source: `src/operators.lean:118`
- Signature: `theorem lifting_comp {a b c: Type} (f: a → b) (g: b → c) (s: stream a) : ↑↑ (λ x, g (f x)) s = ↑↑ g (↑↑ f s)`
- Description: Gives a composition law for `lifting`.

### `lifting2_comp`

- Kind: `theorem`
- Source: `src/operators.lean:121`
- Signature: `theorem lifting2_comp {a b c d e: Type} (f: a → c) (g: b → d) (T: c → d → e) (s1: stream a) (s2: stream b) : ↑² (λ x y, T (f x) (g y)) s1 s2 = ↑²T (↑↑f s1) (↑↑g s2)`
- Description: Gives a composition law for `lifting2`.

### `lifting2_comp'`

- Kind: `theorem`
- Source: `src/operators.lean:126`
- Signature: `theorem lifting2_comp' {a b c d e: Type} (f: a → c) (g: b → d) (T: c → d → e) : ↑² (λ x y, T (f x) (g y)) = λ s1 s2, ↑²T (↑↑f s1) (↑↑g s2)`
- Description: Provides a supporting lemma about doubly lifted operators and their incremental forms.

### `delay`

- Kind: `def`
- Source: `src/operators.lean:134`
- Signature: `def delay : operator a a`
- Description: Defines one-step delay on streams, inserting zero at time `0`.

### `time_invariant`

- Kind: `def`
- Source: `src/operators.lean:147`
- Signature: `def time_invariant (S: operator a b)`
- Description: Defines time invariance for stream operators.

### `time_invariant_comp`

- Kind: `lemma`
- Source: `src/operators.lean:155`
- Signature: `lemma time_invariant_comp (S: operator a b) : time_invariant S ↔ S ∘ z⁻¹ = z⁻¹ ∘ S`
- Description: Gives a composition law for `time_invariant`.

### `lifting_time_invariance`

- Kind: `theorem`
- Source: `src/operators.lean:167`
- Signature: `theorem lifting_time_invariance (f: a → b) : time_invariant (lifting f) ↔ f 0 = 0`
- Description: Shows that `lifting` is time-invariant.

### `lifting_time_invariant`

- Kind: `lemma`
- Source: `src/operators.lean:184`
- Signature: `lemma lifting_time_invariant (f: a → b) : f 0 = 0 → time_invariant (↑↑ f)`
- Description: Shows that `lifting` is time-invariant.

### `delay_t_0`

- Kind: `lemma`
- Source: `src/operators.lean:190`
- Signature: `lemma delay_t_0 (s: stream a) : z⁻¹ s 0 = 0`
- Description: Shows the zero-case behavior of `delay_t`.

### `delay_0`

- Kind: `lemma`
- Source: `src/operators.lean:194`
- Signature: `lemma delay_0 : z⁻¹ (0 : stream a) = 0`
- Description: Shows the zero-case behavior of `delay`.

### `delay_succ`

- Kind: `lemma`
- Source: `src/operators.lean:200`
- Signature: `lemma delay_succ (s: stream a) (n: ℕ) : z⁻¹ s n.succ = s n`
- Description: Gives the successor-step form of `delay`.

### `delay_sub_1`

- Kind: `lemma`
- Source: `src/operators.lean:203`
- Signature: `lemma delay_sub_1 (s: stream a) (n: ℕ) : 0 < n → z⁻¹ s n = s (n-1)`
- Description: Provides a supporting lemma about the delay operator in the stated setting.

### `delay_eq_at`

- Kind: `lemma`
- Source: `src/operators.lean:210`
- Signature: `lemma delay_eq_at (s1 s2: stream a) (t: ℕ) : (0 < t → s1 (t-1) = s2 (t-1)) → z⁻¹ s1 t = z⁻¹ s2 t`
- Description: Specializes `delay_eq` to a concrete time or element index.

### `time_invariant_0_0`

- Kind: `lemma`
- Source: `src/operators.lean:219`
- Signature: `lemma time_invariant_0_0 (S: operator a b) : time_invariant S → S 0 0 = 0`
- Description: Shows the zero-case behavior of `time_invariant_0`.

### `time_invariant_t`

- Kind: `lemma`
- Source: `src/operators.lean:230`
- Signature: `lemma time_invariant_t {S: operator a b} (h: time_invariant S) : ∀ s t, S (delay s) t = delay (S s) t`
- Description: Provides a supporting lemma about `time_invariant_t`.

### `time_invariant_zpp`

- Kind: `lemma`
- Source: `src/operators.lean:240`
- Signature: `lemma time_invariant_zpp (S: operator a b) : time_invariant S → S 0 = 0`
- Description: Establishes the zero-at-zero property for `time_invariant`.

### `lift_time_invariant`

- Kind: `lemma`
- Source: `src/operators.lean:254`
- Signature: `lemma lift_time_invariant (f: a → b) : f 0 = 0 → time_invariant ↑↑f`
- Description: Shows that `lift` is time-invariant.

### `delay_time_invariant`

- Kind: `lemma`
- Source: `src/operators.lean:265`
- Signature: `lemma delay_time_invariant : time_invariant (@delay a _)`
- Description: Shows that `delay` is time-invariant.

### `lifting2_time_invariant`

- Kind: `theorem`
- Source: `src/operators.lean:269`
- Signature: `theorem lifting2_time_invariant (f: a → b → c) : time_invariant (uncurry_op (↑² f)) ↔ f 0 0 = 0`
- Description: Shows that `lifting2` is time-invariant.

### `causal`

- Kind: `def`
- Source: `src/operators.lean:289`
- Signature: `def causal (S: operator a b)`
- Description: Defines causality for stream operators.

### `lifting_causal`

- Kind: `theorem`
- Source: `src/operators.lean:293`
- Signature: `theorem lifting_causal (f: a → b) : causal (lifting f)`
- Description: Shows that `lifting` is causal.

### `delay_causal`

- Kind: `theorem`
- Source: `src/operators.lean:301`
- Signature: `theorem delay_causal : causal (@delay a _)`
- Description: Shows that `delay` is causal.

### `causal_comp_causal`

- Kind: `theorem`
- Source: `src/operators.lean:310`
- Signature: `theorem causal_comp_causal (S1: operator a b) (h1: causal S1) (S2: operator b c) (h2: causal S2) : causal (λ s, S2 (S1 s))`
- Description: Shows that `causal_comp` is causal.

### `causal_respects_agree_upto`

- Kind: `lemma`
- Source: `src/operators.lean:322`
- Signature: `lemma causal_respects_agree_upto (S: operator a b) (h: causal S) (s1 s2: stream a) (n: ℕ) : s1 ==n== s2 → S s1 ==n== S s2`
- Description: Provides a supporting lemma about causality in the stated setting.

### `causal_to_agree`

- Kind: `lemma`
- Source: `src/operators.lean:332`
- Signature: `lemma causal_to_agree (S: operator a b) : causal S ↔ (∀ s1 s2 n, s1 ==n== s2 → S s1 ==n== S s2)`
- Description: Provides a supporting lemma about causality in the stated setting.

### `causal2`

- Kind: `lemma`
- Source: `src/operators.lean:348`
- Signature: `lemma causal2 (T: operator2 a b c) : causal (uncurry_op T) ↔ (∀ s1 s1' s2 s2' n, s1 ==n== s1' → s2 ==n== s2' → T s1 s2 n = T s1' s2' n)`
- Description: Provides a supporting lemma about `causal2`.

### `causal2_agree`

- Kind: `lemma`
- Source: `src/operators.lean:374`
- Signature: `lemma causal2_agree (T: operator2 a b c) : causal (uncurry_op T) → (∀ s1 s1' s2 s2' n, s1 ==n== s1' → s2 ==n== s2' → T s1 s2 ==n== T s1' s2')`
- Description: Provides a supporting lemma about `causal2_agree`.

### `uncurry_op_lifting`

- Kind: `theorem`
- Source: `src/operators.lean:386`
- Signature: `theorem uncurry_op_lifting {d:Type} [add_comm_group d] (f: c → d) (t: stream a → stream b → stream c) : uncurry_op (λ (x: stream a) (y: stream b), ↑↑f (t x y)) = ↑↑f ∘ uncurry_op t`
- Description: Provides a supporting theorem about `uncurry_op_lifting`.

### `causal_uncurry_op_fixed`

- Kind: `lemma`
- Source: `src/operators.lean:393`
- Signature: `lemma causal_uncurry_op_fixed (T: operator2 a b b) : causal (uncurry_op T) → ∀ s, causal (T s)`
- Description: Provides a supporting lemma about causality in the stated setting.

### `lifting_lifting2_comp`

- Kind: `lemma`
- Source: `src/operators.lean:403`
- Signature: `lemma lifting_lifting2_comp {d: Type} [has_zero d] (f: c → d) (g: a → b → c) : ∀ s1 s2, ↑↑f (↑²g s1 s2) = ↑²(λ x y, f (g x y)) s1 s2`
- Description: Gives a composition law for `lifting_lifting2`.

### `uncurry_op_lifting2`

- Kind: `lemma`
- Source: `src/operators.lean:409`
- Signature: `lemma uncurry_op_lifting2 (f: a → b → c) : uncurry_op (↑²f) = ↑ (λ (xy: a × b), f xy.1 xy.2)`
- Description: Provides a supporting lemma about `uncurry_op_lifting2`.

### `strict`

- Kind: `def`
- Source: `src/operators.lean:415`
- Signature: `def strict (S: operator a b)`
- Description: Defines strictness, meaning the next output depends only on earlier input prefixes.

### `strict_unique_zero`

- Kind: `theorem`
- Source: `src/operators.lean:422`
- Signature: `theorem strict_unique_zero (S: operator a b) (h: strict S) : ∀ s s', S s 0 = S s' 0`
- Description: Shows the zero-case behavior of `strict_unique`.

### `strict_causal_to_causal`

- Kind: `theorem`
- Source: `src/operators.lean:432`
- Signature: `theorem strict_causal_to_causal (S: operator a b) : strict S → causal S`
- Description: Shows that `strict_causal_to` is causal.

### `delay_strict`

- Kind: `theorem`
- Source: `src/operators.lean:441`
- Signature: `theorem delay_strict : strict (@delay a _)`
- Description: Shows that `delay` is strict.

### `causal_strict_strict`

- Kind: `theorem`
- Source: `src/operators.lean:449`
- Signature: `theorem causal_strict_strict (F: operator a b) (hstrict: strict F) (T: operator b c) (hcausal: causal T) : strict (λ α, T (F α))`
- Description: Shows that `causal_strict` is strict.

### `strict_causal_strict`

- Kind: `theorem`
- Source: `src/operators.lean:463`
- Signature: `theorem strict_causal_strict (F: operator a b) (hcausal: causal F) (T: operator b c) (hstrict: strict T) : strict (λ α, T (F α))`
- Description: Shows that `strict_causal` is strict.

### `nth`

- Kind: `private def`
- Source: `src/operators.lean:480`
- Signature: `private def nth (F: operator a a) : ℕ → stream a -- We apply F at the bottom so that fix F 0 is given by F rather than being -- forced to be (0 : a). This seems to generalize the paper, which doesn't -- consider such operators! (The assumption that everything is time invariant -- forces operators to have F 0 0 = 0, as proven in [time_invariant_0_0].) | nat.zero`
- Description: Defines the iterated approximation chain used to construct fixpoints of strict operators.

### `nth_0`

- Kind: `lemma`
- Source: `src/operators.lean:489`
- Signature: `lemma nth_0 (F: operator a a) : nth F 0 = F 0`
- Description: Shows the zero-case behavior of `nth`.

### `nth_succ`

- Kind: `lemma`
- Source: `src/operators.lean:492`
- Signature: `lemma nth_succ (F: operator a a) (n: ℕ) : nth F n.succ = F (nth F n)`
- Description: Gives the successor-step form of `nth`.

### `fix`

- Kind: `def`
- Source: `src/operators.lean:502`
- Signature: `def fix (F: operator a a) : stream a`
- Description: Defines the least fixed stream generated by iterating a strict operator from zero.

### `fix_0`

- Kind: `lemma`
- Source: `src/operators.lean:506`
- Signature: `lemma fix_0 (F: operator a a) : fix F 0 = F 0 0`
- Description: Shows the zero-case behavior of `fix`.

### `strict_zpp_zero`

- Kind: `lemma`
- Source: `src/operators.lean:508`
- Signature: `lemma strict_zpp_zero (S: operator a b) (hstrict: strict S) (hzpp: S 0 0 = 0) : ∀ s, S s 0 = 0`
- Description: Shows the zero-case behavior of `strict_zpp`.

### `strict_agree_at_next`

- Kind: `lemma`
- Source: `src/operators.lean:519`
- Signature: `lemma strict_agree_at_next (S: operator a b) (hstrict: strict S) : ∀ s s' n, agree_upto n s s' → S s n.succ = S s' n.succ`
- Description: Provides a supporting lemma about strictness in the stated setting.

### `agree_upto_strict_extend`

- Kind: `lemma`
- Source: `src/operators.lean:529`
- Signature: `lemma agree_upto_strict_extend (S: operator a b) (hstrict: strict S) (s s': stream a) : ∀ n, s ==n== s' → S s ==n.succ== S s'`
- Description: Extends `agree_upto_strict` by one additional time step.

### `strict_as_agree_upto`

- Kind: `lemma`
- Source: `src/operators.lean:543`
- Signature: `lemma strict_as_agree_upto (S: operator a b) : strict S ↔ ((∀ s s', S s ==0== S s') ∧ ∀ s s' n, s ==n== s' → S s ==n.succ== S s')`
- Description: Provides a supporting lemma about strictness in the stated setting.

### `delay_succ_upto`

- Kind: `lemma`
- Source: `src/operators.lean:572`
- Signature: `lemma delay_succ_upto (s1 s2: stream a) (n: ℕ) : s1 ==n== s2 → delay s1 ==n.succ== delay s2`
- Description: Provides a supporting lemma about the delay operator in the stated setting.

### `and_wlog2`

- Kind: `private lemma`
- Source: `src/operators.lean:584`
- Signature: `private lemma and_wlog2 {p1 p2: Prop} (h2: p2) (h21: p2 → p1) : p1 ∧ p2`
- Description: Internal helper: provides a supporting lemma about `and_wlog2`.

### `nth_fix_agree_aux`

- Kind: `private lemma`
- Source: `src/operators.lean:587`
- Signature: `private lemma nth_fix_agree_aux (F: operator a a) (hstrict: strict F) (n: ℕ) : nth F n ==n== fix F ∧ fix F ==n== F (fix F)`
- Description: Internal helper: describes one stage of the approximation sequence `nth`.

### `fix_eq`

- Kind: `theorem`
- Source: `src/operators.lean:621`
- Signature: `theorem fix_eq (F: operator a a) (hstrict: strict F) : fix F = F (fix F)`
- Description: Shows that `fix` is indeed a fixed point of a strict operator.

### `fixpoints_unique`

- Kind: `protected theorem`
- Source: `src/operators.lean:635`
- Signature: `protected theorem fixpoints_unique (F: operator a a) (hstrict: strict F) (α β: stream a) : -- α and β are two possible solutions α = F α → β = F β → α = β`
- Description: Packages the uniqueness result for ordinary strict fixpoints.

### `fix_unique`

- Kind: `theorem`
- Source: `src/operators.lean:652`
- Signature: `theorem fix_unique (F: operator a a) (hstrict: strict F) (α: stream a) (h_fix: α = F α) : α = fix F`
- Description: Shows uniqueness of the fixed point produced by `fix`.

### `fix2`

- Kind: `def`
- Source: `src/operators.lean:663`
- Signature: `def fix2 (F: operator (stream a) (stream a)) : stream (stream a)`
- Description: Defines the nested-stream analogue of `fix` for operators over streams of streams.

### `causal_nested`

- Kind: `def`
- Source: `src/operators.lean:666`
- Signature: `def causal_nested (Q: operator (stream a) (stream b))`
- Description: Defines causality for operators over nested streams.

### `strict2`

- Kind: `def`
- Source: `src/operators.lean:671`
- Signature: `def strict2 (Q: operator (stream a) (stream b))`
- Description: Defines strictness for operators over nested streams.

### `strict2_is_causal_nested`

- Kind: `theorem`
- Source: `src/operators.lean:676`
- Signature: `theorem strict2_is_causal_nested (Q: operator (stream a) (stream b)) : strict2 Q → causal_nested Q`
- Description: Lifts the corresponding property to the nested-stream setting for `strict2_is_causal`.

### `strict2_agree_0`

- Kind: `lemma`
- Source: `src/operators.lean:686`
- Signature: `lemma strict2_agree_0 (F: operator (stream a) (stream b)) (hstrict: strict2 F) : ∀ s s' n, F s n 0 = F s' n 0`
- Description: Shows the zero-case behavior of `strict2_agree`.

### `strict2_eq_0`

- Kind: `lemma`
- Source: `src/operators.lean:695`
- Signature: `lemma strict2_eq_0 (F: operator (stream a) (stream b)) (hstrict: strict2 F) : ∀ s n, F s n 0 = F 0 n 0`
- Description: Shows the zero-case behavior of `strict2_eq`.

### `agree_upto2`

- Kind: `def`
- Source: `src/operators.lean:701`
- Signature: `def agree_upto2 (t: ℕ) (s1 s2: stream (stream a))`
- Description: Defines agreement up to a finite outer time for streams of streams.

### `agree_upto2_symm`

- Kind: `lemma`
- Source: `src/operators.lean:705`
- Signature: `lemma agree_upto2_symm (t: ℕ) (s1 s2: stream (stream a)) : s1 ==t== s2 → s2 ==t== s1`
- Description: Proves symmetry of `agree_upto2`.

### `agree_upto2_trans`

- Kind: `lemma`
- Source: `src/operators.lean:713`
- Signature: `lemma agree_upto2_trans (t: ℕ) (s1 s2 s3: stream (stream a)) : s1 ==t== s2 → s2 ==t== s3 → s1 ==t== s3`
- Description: Proves transitivity of `agree_upto2`.

### `agree_upto2_0`

- Kind: `lemma`
- Source: `src/operators.lean:722`
- Signature: `lemma agree_upto2_0 (s1 s2: stream (stream a)) : s1 ==0== s2 ↔ (∀ n, s1 n 0 = s2 n 0)`
- Description: Shows the zero-case behavior of `agree_upto2`.

### `agree_upto2_extend`

- Kind: `lemma`
- Source: `src/operators.lean:729`
- Signature: `lemma agree_upto2_extend (t: ℕ) (s s': stream (stream a)) : s ==t== s' → (∀ n, s n t.succ = s' n t.succ) → s ==t.succ== s'`
- Description: Extends `agree_upto2` by one additional time step.

### `agree_upto2_strict_extend`

- Kind: `lemma`
- Source: `src/operators.lean:742`
- Signature: `lemma agree_upto2_strict_extend (S: operator (stream a) (stream b)) (hstrict: strict2 S) (s s': stream (stream a)) : ∀ t, s ==t== s' → S s ==t.succ== S s'`
- Description: Extends `agree_upto2_strict` by one additional time step.

### `strict_agree2_at_next`

- Kind: `lemma`
- Source: `src/operators.lean:755`
- Signature: `lemma strict_agree2_at_next (S: operator (stream a) (stream b)) (hstrict: strict2 S) : ∀ s s' t, s ==t== s' → ∀ n, S s n t.succ = S s' n t.succ`
- Description: Provides a supporting lemma about strictness in the stated setting.

### `nth_fix2_agree_aux`

- Kind: `private lemma`
- Source: `src/operators.lean:764`
- Signature: `private lemma nth_fix2_agree_aux (F: operator (stream a) (stream a)) (hstrict: strict2 F) (t: ℕ) : nth F t ==t== fix2 F ∧ fix2 F ==t== F (fix2 F)`
- Description: Internal helper: describes one stage of the approximation sequence `nth`.

### `fix2_eq`

- Kind: `theorem`
- Source: `src/operators.lean:795`
- Signature: `theorem fix2_eq (F: operator (stream a) (stream a)) (hstrict: strict2 F) : fix2 F = F (fix2 F)`
- Description: Shows that `fix2` is a fixed point in the nested-stream setting.

### `agree2_everywhere_eq`

- Kind: `theorem`
- Source: `src/operators.lean:804`
- Signature: `theorem agree2_everywhere_eq (s1 s2: stream (stream a)) : (∀ t, s1 ==t== s2) → s1 = s2`
- Description: Shows that agreement everywhere implies actual equality for `agree2`.

### `fixpoints2_unique`

- Kind: `protected theorem`
- Source: `src/operators.lean:812`
- Signature: `protected theorem fixpoints2_unique (F: operator (stream a) (stream a)) (hstrict: strict2 F) (α β: stream (stream a)) : -- α and β are two possible solutions α = F α → β = F β → α = β`
- Description: Packages the uniqueness result for nested strict fixpoints.

### `fix2_unique`

- Kind: `theorem`
- Source: `src/operators.lean:830`
- Signature: `theorem fix2_unique (F: operator (stream a) (stream a)) (hstrict: strict2 F) (α: stream (stream a)) (h_fix: α = F α) : α = fix2 F`
- Description: Shows uniqueness of the nested-stream fixed point produced by `fix2`.

### `lifting_delay_strict2`

- Kind: `theorem`
- Source: `src/operators.lean:841`
- Signature: `theorem lifting_delay_strict2 : strict2 (↑↑ (@delay a _))`
- Description: Provides a supporting lemma about lifted operators.

### `causal_nested_const`

- Kind: `theorem`
- Source: `src/operators.lean:851`
- Signature: `theorem causal_nested_const (c: stream (stream b)) : causal_nested (λ (x: stream (stream a)), c)`
- Description: Provides a supporting lemma about causality in the stated setting.

### `causal_nested_id`

- Kind: `theorem`
- Source: `src/operators.lean:857`
- Signature: `theorem causal_nested_id : causal_nested (λ (x: stream (stream a)), x)`
- Description: Provides a supporting lemma about causality in the stated setting.

### `causal_nested_comp`

- Kind: `theorem`
- Source: `src/operators.lean:863`
- Signature: `theorem causal_nested_comp (Q1: operator (stream b) (stream c)) (Q2: operator (stream a) (stream b)) : causal_nested Q1 → causal_nested Q2 → causal_nested (λ s, Q1 (Q2 s))`
- Description: Gives a composition law for `causal_nested`.

### `causal_nested_lifting`

- Kind: `theorem`
- Source: `src/operators.lean:877`
- Signature: `theorem causal_nested_lifting (Q: operator a b) : causal Q → causal_nested (↑↑Q)`
- Description: Provides a supporting lemma about causality in the stated setting.

### `feedback_ckt_body_strict`

- Kind: `theorem`
- Source: `src/operators.lean:889`
- Signature: `theorem feedback_ckt_body_strict (F: operator b b) (hstrict: strict F) (T: operator2 a b b) (hcausal: causal (uncurry_op T)) (s: stream a) : strict (λ α, T s (F α))`
- Description: Shows strictness of the operator body used by circuit feedback.

### `feedback_ckt_unfold`

- Kind: `lemma`
- Source: `src/operators.lean:899`
- Signature: `lemma feedback_ckt_unfold (F: operator b b) (hstrict: strict F) (T: operator2 a b b) (hcausal: causal (uncurry_op T)) (s: stream a) : fix (λ α, T s (F α)) = T s (F (fix (λ α, T s (F α))))`
- Description: Unfolds the denotation of circuit feedback.

### `feedback_ckt_causal`

- Kind: `theorem`
- Source: `src/operators.lean:908`
- Signature: `theorem feedback_ckt_causal (F: operator b b) (hstrict: strict F) (T: operator2 a b b) (hcausal: causal (uncurry_op T)) : causal (λ s, fix (λ α, T s (F α)))`
- Description: Shows that circuit feedback denotes a causal operator.

## `ordering.lean`

Order-theoretic properties of streams such as positivity and monotonicity, especially as they interact with derivative and integral.

Declarations in this module: **7**.

### `positive`

- Kind: `def`
- Source: `src/ordering.lean:8`
- Signature: `def positive (s: stream a)`
- Description: Defines positivity of a stream in the pointwise order.

### `stream_monotone`

- Kind: `def`
- Source: `src/ordering.lean:9`
- Signature: `def stream_monotone (s: stream a)`
- Description: Defines monotonic growth of a stream over time.

### `is_positive`

- Kind: `def`
- Source: `src/ordering.lean:10`
- Signature: `def is_positive {b: Type} [ordered_add_comm_group b] (f: stream a → stream b)`
- Description: Defines positivity preservation for operators on streams.

### `stream_monotone_order`

- Kind: `theorem`
- Source: `src/ordering.lean:20`
- Signature: `theorem stream_monotone_order (s: stream a) : stream_monotone s ↔ (∀ t1 t2, t1 ≤ t2 → s t1 ≤ s t2)`
- Description: Characterizes stream monotonicity as ordinary order preservation over time indices.

### `integral_monotone`

- Kind: `lemma`
- Source: `src/ordering.lean:36`
- Signature: `lemma integral_monotone (s: stream a) : positive s → stream_monotone (I s)`
- Description: Shows that integrating a positive stream yields a monotone stream.

### `derivative_pos`

- Kind: `lemma`
- Source: `src/ordering.lean:47`
- Signature: `lemma derivative_pos (s: stream a) : -- NOTE: paper is missing this, but it is also necessary (maybe they -- intend `s[-1] =0` in the definition of monotone) 0 ≤ s 0 → stream_monotone s → positive (D s)`
- Description: Shows that the derivative of a monotone stream is positive when the initial value is nonnegative.

### `derivative_pos_counter_example`

- Kind: `lemma`
- Source: `src/ordering.lean:63`
- Signature: `lemma derivative_pos_counter_example : (∃ (x:a), x < 0) → ¬(∀ (s: stream a), stream_monotone s → positive (D s))`
- Description: Provides a counterexample showing that monotonicity alone does not guarantee positivity of the derivative without an initial-value side condition.

## `recursive.lean`

Recursive DBSP semantics, including a fixpoint construction and the equivalence between naive and seminaive recursion.

Declarations in this module: **13**.

### `approxs`

- Kind: `private def`
- Source: `src/recursive.lean:19`
- Signature: `private def approxs : stream Z[a]`
- Description: Defines the stream of successive recursive approximants obtained by repeated application of a recursive body.

### `approxs_unfold`

- Kind: `lemma`
- Source: `src/recursive.lean:22`
- Signature: `lemma approxs_unfold : approxs R = ↑↑R (z⁻¹ (approxs R))`
- Description: Unfolds the recursive approximant stream into one delayed application of the recursive body.

### `recursive_fixpoint`

- Kind: `noncomputable def`
- Source: `src/recursive.lean:31`
- Signature: `noncomputable def recursive_fixpoint : Z[a]`
- Description: Defines the scalar recursive fixpoint by integrating the derivative of the approximant stream.

### `approxs_apply`

- Kind: `lemma`
- Source: `src/recursive.lean:34`
- Signature: `lemma approxs_apply (n: ℕ) : approxs R n = (R^[n.succ]) 0`
- Description: Identifies the `n`th approximant with the `(n+1)`-fold iterate of the recursive body starting from zero.

### `approxs_unfold_succ`

- Kind: `lemma`
- Source: `src/recursive.lean:46`
- Signature: `lemma approxs_unfold_succ (n: ℕ) : approxs R n.succ = R (approxs R n)`
- Description: Specializes the approximant unfolding law to successor times.

### `eq_succ_is_fixpoint`

- Kind: `private lemma`
- Source: `src/recursive.lean:56`
- Signature: `private lemma eq_succ_is_fixpoint (n: ℕ) (heqn: R^[n.succ] 0 = (R^[n]) 0) : ∀ m ≥ n, R^[m] 0 = (R^[n]) 0`
- Description: Internal helper: provides a supporting lemma about `eq_succ_is_fixpoint`.

### `derivative_approx_almost_zero`

- Kind: `lemma`
- Source: `src/recursive.lean:77`
- Signature: `lemma derivative_approx_almost_zero (n: ℕ) (heqn: (R^[n.succ]) 0 = (R^[n]) 0) : zero_after (D (approxs R)) n.succ`
- Description: Shows that once the recursive iterates stabilize, the derivative of the approximant stream is eventually zero.

### `recursive_fixpoint_ok`

- Kind: `theorem`
- Source: `src/recursive.lean:90`
- Signature: `theorem recursive_fixpoint_ok (n: ℕ) (heqn: (R^[n.succ]) 0 = (R^[n]) 0) : recursive_fixpoint R = (R^[n]) 0`
- Description: Shows that the recursive fixpoint equals the stabilized iterate once a fixed point has been reached.

### `naive`

- Kind: `noncomputable def`
- Source: `src/recursive.lean:113`
- Signature: `noncomputable def naive : Z[b] → Z[a]`
- Description: Defines the naive recursive evaluation strategy for a binary recursive body.

### `seminaive`

- Kind: `noncomputable def`
- Source: `src/recursive.lean:116`
- Signature: `noncomputable def seminaive : Z[b] → Z[a]`
- Description: Defines the seminaive recursive evaluation strategy.

### `seminaive_equiv`

- Kind: `theorem`
- Source: `src/recursive.lean:122`
- Signature: `theorem seminaive_equiv : seminaive R = naive R`
- Description: Shows that seminaive recursion computes the same result as the naive recursive formulation.

### `naive_ok`

- Kind: `theorem`
- Source: `src/recursive.lean:137`
- Signature: `theorem naive_ok (i: Z[b]) (n: ℕ) (heqn: (R i)^[n.succ] 0 = ((R i)^[n]) 0) : naive R i = ((R i)^[n]) 0`
- Description: Proves correctness of `naive` when the recursive body stabilizes after finitely many iterations.

### `seminaive_ok`

- Kind: `theorem`
- Source: `src/recursive.lean:151`
- Signature: `theorem seminaive_ok (i: Z[b]) (n: ℕ) (heqn: (R i)^[n.succ] 0 = ((R i)^[n]) 0) : seminaive R i = ((R i)^[n]) 0`
- Description: Transfers the naive correctness result to `seminaive` via their equivalence.

## `recursive_example.lean`

A worked transitive-closure example that derives increasingly optimized recursive and incremental formulations.

Declarations in this module: **35**.

### `Edge`

- Kind: `def`
- Source: `src/recursive_example.lean:12`
- Signature: `def Edge (Node: Type)`
- Description: Defines the edge type used in the transitive-closure example.

### `πh`

- Kind: `def`
- Source: `src/recursive_example.lean:17`
- Signature: `def πh (input: Edge Node) : Edge Node`
- Description: Projects an edge to a self-loop on its head vertex.

### `πt`

- Kind: `def`
- Source: `src/recursive_example.lean:19`
- Signature: `def πt (input: Edge Node) : Edge Node`
- Description: Projects an edge to a self-loop on its tail vertex.

### `πht`

- Kind: `def`
- Source: `src/recursive_example.lean:21`
- Signature: `def πht (x: Edge Node × Edge Node) : Edge Node`
- Description: Recombines a recursive path edge with an input edge into a longer path edge.

### `closure1`

- Kind: `def`
- Source: `src/recursive_example.lean:28`
- Signature: `def closure1 (E R1: Z[Edge Node]) : Z[Edge Node]`
- Description: Defines one recursive closure step for the transitive-closure example.

### `lifting_closure1_eq`

- Kind: `lemma`
- Source: `src/recursive_example.lean:35`
- Signature: `lemma lifting_closure1_eq (E R1: stream Z[Edge Node]) : ↑²closure1 E R1 = ↑distinct ( ↑(zset.map πht) (↑²(equi_join prod.snd prod.fst) R1 E) + E + ↑(zset.map πh) E + ↑(zset.map πt) E)`
- Description: Expands the lifted form of `closure1` into the stream-operator expression used in later rewrites.

### `closure`

- Kind: `noncomputable def`
- Source: `src/recursive_example.lean:43`
- Signature: `noncomputable def closure : Z[Edge Node] → Z[Edge Node]`
- Description: Defines transitive closure via the generic naive recursive construction.

### `closure_seminaive`

- Kind: `noncomputable def`
- Source: `src/recursive_example.lean:45`
- Signature: `noncomputable def closure_seminaive : Z[Edge Node] → Z[Edge Node]`
- Description: Defines the seminaive transitive-closure formulation.

### `closure_efficient_ok`

- Kind: `theorem`
- Source: `src/recursive_example.lean:53`
- Signature: `theorem closure_efficient_ok : @closure_seminaive Node _ = closure`
- Description: Shows that the seminaive closure implementation is equivalent to the naive recursive closure.

### `incremental_closure`

- Kind: `noncomputable def`
- Source: `src/recursive_example.lean:74`
- Signature: `noncomputable def incremental_closure : operator (Z[Edge Node]) (Z[Edge Node])`
- Description: Defines the first incrementalized transitive-closure operator.

### `incremental_closure_ok`

- Kind: `theorem`
- Source: `src/recursive_example.lean:86`
- Signature: `theorem incremental_closure_ok : @incremental_closure Node _ = (↑closure)^Δ`
- Description: Shows that `incremental_closure` computes the incrementalized form of `closure`.

### `incremental_closure2`

- Kind: `noncomputable def`
- Source: `src/recursive_example.lean:111`
- Signature: `noncomputable def incremental_closure2 : operator Z[Edge Node] Z[Edge Node]`
- Description: Defines a second, more aggressively incrementalized closure operator.

### `fix2_congr`

- Kind: `lemma`
- Source: `src/recursive_example.lean:121`
- Signature: `lemma fix2_congr {a: Type} [add_comm_group a] (F1 F2: operator (stream a) (stream a)) : F1 = F2 → fix2 F1 = fix2 F2`
- Description: Shows that equal nested-stream operators yield equal `fix2` results.

### `lifting_map_πht_incremental`

- Kind: `lemma`
- Source: `src/recursive_example.lean:126`
- Signature: `lemma lifting_map_πht_incremental : ↑(↑(zset.map (@πht Node _)))^Δ = ↑(↑(zset.map πht))`
- Description: Shows that lifting `map πht` commutes with incrementalization in the expected way.

### `incremental_closure2_ok`

- Kind: `theorem`
- Source: `src/recursive_example.lean:132`
- Signature: `theorem incremental_closure2_ok : @incremental_closure2 Node _ = (↑closure)^Δ`
- Description: Shows that the more aggressively incrementalized closure operator is still the incrementalization of `closure`.

### `distinct_double_incremental`

- Kind: `def`
- Source: `src/recursive_example.lean:176`
- Signature: `def distinct_double_incremental {A: Type} [decidable_eq A] : operator (stream Z[A]) (stream Z[A])`
- Description: Defines the double-incremental form of `distinct` used in the recursive example.

### `distinct_double_incremental_ok`

- Kind: `theorem`
- Source: `src/recursive_example.lean:179`
- Signature: `theorem distinct_double_incremental_ok {A: Type} [decidable_eq A] : (↑(↑(@distinct A _)^Δ))^Δ = distinct_double_incremental`
- Description: Shows that double incrementalization of `distinct` matches the explicit operator `distinct_double_incremental`.

### `join_double_incremental1`

- Kind: `def`
- Source: `src/recursive_example.lean:201`
- Signature: `def join_double_incremental1 : operator2 (stream (Z[A])) (stream Z[B]) (stream Z[A × B])`
- Description: Defines an intermediate double-incremental join plan.

### `lifting_I_delay_incremental`

- Kind: `lemma`
- Source: `src/recursive_example.lean:208`
- Signature: `lemma lifting_I_delay_incremental {a: Type} [add_comm_group a] : incremental ↑(λ (x: stream a), I (z⁻¹ x)) = ↑(λ x, I (z⁻¹ x))`
- Description: Provides a supporting lemma about lifted operators.

### `lifting_I_delay_simplify`

- Kind: `lemma`
- Source: `src/recursive_example.lean:217`
- Signature: `lemma lifting_I_delay_simplify (s: stream (stream Z[A])) : ↑(λ x, I (z⁻¹ x)) s = ↑z⁻¹ (↑I s)`
- Description: Provides a supporting lemma about lifted operators.

### `join_double_incremental1_ok`

- Kind: `theorem`
- Source: `src/recursive_example.lean:225`
- Signature: `theorem join_double_incremental1_ok : ↑²(↑²(equi_join π1 π2)^Δ2)^Δ2 = join_double_incremental1 π1 π2`
- Description: Shows that one double-incremental join expansion matches `join_double_incremental1`.

### `join_double_incremental`

- Kind: `def`
- Source: `src/recursive_example.lean:242`
- Signature: `def join_double_incremental : operator2 (stream Z[A]) (stream Z[B]) (stream Z[A × B])`
- Description: Defines the fully optimized double-incremental join plan.

### `equi_join_lifting2_time_invariant`

- Kind: `lemma`
- Source: `src/recursive_example.lean:250`
- Signature: `lemma equi_join_lifting2_time_invariant : ∀ s1 s2, z⁻¹ (↑²(↑² (equi_join π1 π2)) s1 s2) = ↑²(↑² (equi_join π1 π2)) (z⁻¹ s1) (z⁻¹ s2)`
- Description: Shows time invariance for the doubly lifted equi-join operator.

### `equi_join_double_lift_bilinear`

- Kind: `lemma`
- Source: `src/recursive_example.lean:259`
- Signature: `lemma equi_join_double_lift_bilinear : bilinear (↑²(↑² (equi_join π1 π2)))`
- Description: Shows bilinearity for the doubly lifted equi-join operator.

### `equi_join_I_1`

- Kind: `lemma`
- Source: `src/recursive_example.lean:267`
- Signature: `lemma equi_join_I_1 : ∀ a b, ↑²(↑² (equi_join π1 π2)) (I a) b = ↑²(↑² (equi_join π1 π2)) a b + ↑²(↑² (equi_join π1 π2)) (z⁻¹ (I a)) b`
- Description: Provides a supporting lemma about equi-join in the stated setting.

### `equi_join_lift_I_1`

- Kind: `lemma`
- Source: `src/recursive_example.lean:279`
- Signature: `lemma equi_join_lift_I_1 : ∀ a b, ↑²(↑² (equi_join π1 π2)) (↑I a) b = ↑²(↑² (equi_join π1 π2)) a b + ↑²(↑² (equi_join π1 π2)) (↑I (↑z⁻¹ a)) b`
- Description: Provides a supporting lemma about equi-join in the stated setting.

### `equi_join_lift_I_2`

- Kind: `lemma`
- Source: `src/recursive_example.lean:295`
- Signature: `lemma equi_join_lift_I_2 : ∀ a b, ↑²(↑² (equi_join π1 π2)) a (↑I b) = ↑²(↑² (equi_join π1 π2)) a b + ↑²(↑² (equi_join π1 π2)) a (↑I (↑z⁻¹ b))`
- Description: Provides a supporting lemma about equi-join in the stated setting.

### `equi_join_I_2`

- Kind: `lemma`
- Source: `src/recursive_example.lean:311`
- Signature: `lemma equi_join_I_2 : ∀ a b, ↑²(↑² (equi_join π1 π2)) a (I b) = ↑²(↑² (equi_join π1 π2)) a b + ↑²(↑² (equi_join π1 π2)) a (z⁻¹ (I b))`
- Description: Provides a supporting lemma about equi-join in the stated setting.

### `equi_join_I_unfold`

- Kind: `lemma`
- Source: `src/recursive_example.lean:323`
- Signature: `lemma equi_join_I_unfold : ∀ a b, ↑²(↑² (equi_join π1 π2)) (I a) (I b) = ↑²(↑² (equi_join π1 π2)) a b + ↑²(↑² (equi_join π1 π2)) a (z⁻¹ (I b)) + ↑²(↑² (equi_join π1 π2)) (z⁻¹ (I a)) b + ↑²(↑² (equi_join π1 π2)) (z⁻¹ (I a)) (z⁻¹ (I b))`
- Description: Unfolds the definition of `equi_join_I`.

### `neg_add_sub`

- Kind: `private lemma`
- Source: `src/recursive_example.lean:335`
- Signature: `private lemma neg_add_sub {α: Type} [add_comm_group α] (x y: α) : (-1 : ℤ) • x + y = y - x`
- Description: Internal helper: provides a supporting lemma about `neg_add_sub`.

### `add_both_sides`

- Kind: `private lemma`
- Source: `src/recursive_example.lean:341`
- Signature: `private lemma add_both_sides {G} [has_add G] [is_right_cancel_add G] (x: G) {a b: G} : a + x = b + x -> a = b`
- Description: Internal helper: provides a supporting lemma about `add_both_sides`.

### `fold_join_helper`

- Kind: `private lemma`
- Source: `src/recursive_example.lean:347`
- Signature: `private lemma fold_join_helper : ∀ a b, ((-1 : ℤ) • (π1▹◃π2) (I (z⁻¹ (↑I a))) b + (π1▹◃π2) (I (z⁻¹ (↑I (↑z⁻¹ a)))) b) = (-1 : ℤ) • (π1▹◃π2) (I (z⁻¹ a)) b`
- Description: Internal helper: provides a supporting lemma about `fold_join_helper`.

### `join_double_incremental_ok`

- Kind: `theorem`
- Source: `src/recursive_example.lean:376`
- Signature: `theorem join_double_incremental_ok : ↑²(↑²(equi_join π1 π2)^Δ2)^Δ2 = join_double_incremental π1 π2`
- Description: Shows that the fully optimized double-incremental join still matches the abstract double incrementalization.

### `incremental_closure_opt`

- Kind: `noncomputable def`
- Source: `src/recursive_example.lean:411`
- Signature: `noncomputable def incremental_closure_opt : operator Z[Edge Node] Z[Edge Node]`
- Description: Defines the optimized incremental transitive-closure operator built from the optimized join kernel.

### `incremental_closure_opt_ok`

- Kind: `theorem`
- Source: `src/recursive_example.lean:421`
- Signature: `theorem incremental_closure_opt_ok : @incremental_closure_opt Node _ = (↑closure)^Δ`
- Description: Shows that the optimized incremental closure operator still implements the incrementalized closure query.

## `relational.lean`

Relational algebra over Z-sets: setification, union, selection, product, joins, intersection, difference, grouping, and distinct-dedup rewrite laws.

Declarations in this module: **63**.

### `distinct_is_set`

- Kind: `lemma`
- Source: `src/relational.lean:12`
- Signature: `lemma distinct_is_set (m: Z[A]) : is_set (distinct m)`
- Description: Provides a supporting lemma about `distinct` in the stated setting.

### `distinct_is_bag`

- Kind: `lemma`
- Source: `src/relational.lean:18`
- Signature: `lemma distinct_is_bag (m: Z[A]) : is_bag (distinct m)`
- Description: Provides a supporting lemma about `distinct` in the stated setting.

### `distinct_set_id`

- Kind: `lemma`
- Source: `src/relational.lean:23`
- Signature: `lemma distinct_set_id (m: Z[A]) : is_set m → m.distinct = m`
- Description: Provides a supporting lemma about `distinct` in the stated setting.

### `distinct_set_simp`

- Kind: `lemma`
- Source: `src/relational.lean:33`
- Signature: `lemma distinct_set_simp (m: Z[A]) : is_set (distinct m) ↔ true`
- Description: Provides a supporting lemma about `distinct` in the stated setting.

### `distinct_bag_simp`

- Kind: `lemma`
- Source: `src/relational.lean:37`
- Signature: `lemma distinct_bag_simp (m: Z[A]) : is_bag (distinct m) ↔ true`
- Description: Provides a supporting lemma about `distinct` in the stated setting.

### `distinct_elem`

- Kind: `lemma`
- Source: `src/relational.lean:40`
- Signature: `lemma distinct_elem {m: Z[A]} {a: A} : is_bag m → (a ∈ m.distinct ↔ a ∈ m)`
- Description: Provides a supporting lemma about `distinct` in the stated setting.

### `distinct_pos`

- Kind: `lemma`
- Source: `src/relational.lean:52`
- Signature: `lemma distinct_pos : fun_positive (@distinct A _)`
- Description: Shows that `distinct` is positive or positivity-preserving.

### `distinct_0`

- Kind: `lemma`
- Source: `src/relational.lean:58`
- Signature: `lemma distinct_0 : distinct (0 : Z[A]) = 0`
- Description: Shows the zero-case behavior of `distinct`.

### `query`

- Kind: `def`
- Source: `src/relational.lean:60`
- Signature: `def query (A B: Type)`
- Description: Defines the type of unary relational queries over Z-sets.

### `union`

- Kind: `def`
- Source: `src/relational.lean:62`
- Signature: `def union (m1 m2: Z[A])`
- Description: Defines set-style union by adding Z-sets and then applying `distinct`.

### `union_eq`

- Kind: `lemma`
- Source: `src/relational.lean:64`
- Signature: `lemma union_eq (m1 m2: Z[A]) : m1 ∪ m2 = union m1 m2`
- Description: States an equality characterizing `union`.

### `union_apply`

- Kind: `lemma`
- Source: `src/relational.lean:66`
- Signature: `lemma union_apply (m1 m2: Z[A]) (a: A) : union m1 m2 a = if 0 < m1 a + m2 a then 1 else 0`
- Description: Gives the pointwise evaluation rule for `union`.

### `union_ok`

- Kind: `theorem`
- Source: `src/relational.lean:74`
- Signature: `theorem union_ok (s1 s2: finset A) : zset.to_set (zset.from_set s1 ∪ zset.from_set s2) = s1 ∪ s2`
- Description: Shows that `union` matches its intended semantics.

### `union_pos`

- Kind: `theorem`
- Source: `src/relational.lean:84`
- Signature: `theorem union_pos : fun_positive2 (@union A _)`
- Description: Shows that `union` is positive or positivity-preserving.

### `map_ok`

- Kind: `theorem`
- Source: `src/relational.lean:92`
- Signature: `theorem map_ok (f: A → B) (s: finset A) : (zset.map f (zset.from_set s)).support = s.image f`
- Description: Shows that `map` matches its intended semantics.

### `map_pos`

- Kind: `theorem`
- Source: `src/relational.lean:100`
- Signature: `theorem map_pos (f: A → B) : fun_positive (zset.map f)`
- Description: Shows that `map` is positive or positivity-preserving.

### `filter`

- Kind: `def`
- Source: `src/relational.lean:111`
- Signature: `def filter (m: Z[A]) : Z[A]`
- Description: Defines relational selection over Z-sets.

### `filter_support`

- Kind: `lemma`
- Source: `src/relational.lean:114`
- Signature: `lemma filter_support (m: Z[A]) : dfinsupp.support (filter p m) = m.support.filter p`
- Description: Characterizes the support of `filter`.

### `filter_apply`

- Kind: `lemma`
- Source: `src/relational.lean:122`
- Signature: `lemma filter_apply (m: Z[A]) (a: A) : filter p m a = if p a then m a else 0`
- Description: Gives the pointwise evaluation rule for `filter`.

### `filter_ok`

- Kind: `theorem`
- Source: `src/relational.lean:129`
- Signature: `theorem filter_ok (s: finset A) : zset.to_set (filter p (zset.from_set s)) = s.filter p`
- Description: Shows that `filter` matches its intended semantics.

### `filter_linear`

- Kind: `lemma`
- Source: `src/relational.lean:137`
- Signature: `lemma filter_linear : ∀ (m1 m2: Z[A]), filter p (m1 + m2) = filter p m1 + filter p m2`
- Description: Shows that `filter` is linear.

### `filter_pos`

- Kind: `theorem`
- Source: `src/relational.lean:147`
- Signature: `theorem filter_pos : fun_positive (filter p)`
- Description: Shows that `filter` is positive or positivity-preserving.

### `filter_0`

- Kind: `lemma`
- Source: `src/relational.lean:155`
- Signature: `lemma filter_0 : filter p 0 = 0`
- Description: Shows the zero-case behavior of `filter`.

### `product`

- Kind: `def`
- Source: `src/relational.lean:160`
- Signature: `def product (m1: Z[A]) (m2: Z[B]) : Z[A × B]`
- Description: Defines Cartesian product over Z-sets.

### `product_apply`

- Kind: `lemma`
- Source: `src/relational.lean:168`
- Signature: `lemma product_apply (m1: Z[A]) (m2: Z[B]) (ab: A × B) : product m1 m2 ab = m1 ab.1 * m2 ab.2`
- Description: Gives the pointwise evaluation rule for `product`.

### `product_ok`

- Kind: `theorem`
- Source: `src/relational.lean:177`
- Signature: `theorem product_ok (s1: finset A) (s2: finset B) : zset.to_set (product (zset.from_set s1) (zset.from_set s2)) = finset.product s1 s2`
- Description: Shows that `product` matches its intended semantics.

### `product_bilinear`

- Kind: `theorem`
- Source: `src/relational.lean:184`
- Signature: `theorem product_bilinear : bilinear (@product A B _ _)`
- Description: Shows that `product` is bilinear.

### `product_pos`

- Kind: `theorem`
- Source: `src/relational.lean:195`
- Signature: `theorem product_pos : fun_positive2 (@product A B _ _)`
- Description: Shows that `product` is positive or positivity-preserving.

### `product_0`

- Kind: `lemma`
- Source: `src/relational.lean:206`
- Signature: `lemma product_0 : @product A B _ _ 0 0 = 0`
- Description: Shows the zero-case behavior of `product`.

### `equi_join`

- Kind: `def`
- Source: `src/relational.lean:212`
- Signature: `def equi_join (m1: Z[A]) (m2: Z[B]) : Z[A × B]`
- Description: Defines equi-join as filtered Cartesian product.

### `equi_join_apply`

- Kind: `lemma`
- Source: `src/relational.lean:216`
- Signature: `lemma equi_join_apply (m1: Z[A]) (m2: Z[B]) (t: A × B) : equi_join π1 π2 m1 m2 t = if π1 t.1 = π2 t.2 then m1 t.1 * m2 t.2 else 0`
- Description: Gives the pointwise evaluation rule for `equi_join`.

### `equi_join_bilinear`

- Kind: `theorem`
- Source: `src/relational.lean:220`
- Signature: `theorem equi_join_bilinear : bilinear (equi_join π1 π2)`
- Description: Shows that `equi_join` is bilinear.

### `equi_join_pos`

- Kind: `theorem`
- Source: `src/relational.lean:227`
- Signature: `theorem equi_join_pos : fun_positive2 (equi_join π1 π2)`
- Description: Shows that `equi_join` is positive or positivity-preserving.

### `equi_join_0_l`

- Kind: `lemma`
- Source: `src/relational.lean:234`
- Signature: `lemma equi_join_0_l (b: Z[B]) : equi_join π1 π2 0 b = 0`
- Description: Shows the left-zero law for `equi_join`.

### `equi_join_0_r`

- Kind: `lemma`
- Source: `src/relational.lean:237`
- Signature: `lemma equi_join_0_r (a: Z[A]) : equi_join π1 π2 a 0 = 0`
- Description: Shows the right-zero law for `equi_join`.

### `intersect`

- Kind: `def`
- Source: `src/relational.lean:252`
- Signature: `def intersect (m1 m2: Z[A]) : Z[A]`
- Description: Defines intersection over Z-sets.

### `intersect_apply`

- Kind: `lemma`
- Source: `src/relational.lean:257`
- Signature: `lemma intersect_apply (m1 m2: Z[A]) (a: A) : (m1 ∩ m2) a = m1 a * m2 a`
- Description: Gives the pointwise evaluation rule for `intersect`.

### `intersect_0`

- Kind: `lemma`
- Source: `src/relational.lean:266`
- Signature: `lemma intersect_0 : (0: Z[A]) ∩ 0 = 0`
- Description: Shows the zero-case behavior of `intersect`.

### `intersect_support`

- Kind: `lemma`
- Source: `src/relational.lean:269`
- Signature: `lemma intersect_support (m1 m2: Z[A]) : (m1 ∩ m2).support = m1.support ∩ m2.support`
- Description: Characterizes the support of `intersect`.

### `intersect_ok`

- Kind: `theorem`
- Source: `src/relational.lean:278`
- Signature: `theorem intersect_ok (s1 s2: finset A) : zset.to_set (zset.from_set s1 ∩ zset.from_set s2) = s1 ∩ s2`
- Description: Shows that `intersect` matches its intended semantics.

### `intersect_pos`

- Kind: `theorem`
- Source: `src/relational.lean:285`
- Signature: `theorem intersect_pos : fun_positive2 ((∩) : Z[A] → Z[A] → Z[A])`
- Description: Shows that `intersect` is positive or positivity-preserving.

### `intersect_bilinear`

- Kind: `theorem`
- Source: `src/relational.lean:293`
- Signature: `theorem intersect_bilinear : bilinear ((∩) : Z[A] → Z[A] → Z[A])`
- Description: Shows that `intersect` is bilinear.

### `difference`

- Kind: `def`
- Source: `src/relational.lean:299`
- Signature: `def difference (m1 m2: Z[A]) : Z[A]`
- Description: Defines set-style difference by subtraction followed by `distinct`.

### `difference_ok`

- Kind: `theorem`
- Source: `src/relational.lean:301`
- Signature: `theorem difference_ok (s1 s2: finset A) : zset.to_set (difference (zset.from_set s1) (zset.from_set s2)) = s1 \ s2`
- Description: Shows that `difference` matches its intended semantics.

### `group_by`

- Kind: `def`
- Source: `src/relational.lean:313`
- Signature: `def group_by : Z[A] → Π₀ (_: K), Z[A]`
- Description: Groups a Z-set by a key function into a finitely supported map of per-key Z-sets.

### `group_by_apply`

- Kind: `lemma`
- Source: `src/relational.lean:319`
- Signature: `lemma group_by_apply (m: Z[A]) (k: K) (a: A) : group_by p m k a = if p a = k then m a else 0`
- Description: Gives the pointwise evaluation rule for `group_by`.

### `group_by_support`

- Kind: `lemma`
- Source: `src/relational.lean:329`
- Signature: `lemma group_by_support (m: Z[A]) (k: K) : (group_by p m k).support = m.support.filter (λ a, p a = k)`
- Description: Characterizes the support of `group_by`.

### `elem_group_by`

- Kind: `lemma`
- Source: `src/relational.lean:335`
- Signature: `lemma elem_group_by (m: Z[A]) (k: K) (a: A) : a ∈ group_by p m k ↔ p a = k ∧ a ∈ m`
- Description: Provides a supporting lemma about `elem_group_by`.

### `group_by_linear`

- Kind: `lemma`
- Source: `src/relational.lean:341`
- Signature: `lemma group_by_linear (m1 m2: Z[A]) : group_by p (m1 + m2) = group_by p m1 + group_by p m2`
- Description: Shows that `group_by` is linear.

### `ite_ite`

- Kind: `lemma`
- Source: `src/relational.lean:353`
- Signature: `lemma ite_ite {A: Type} {c1: Prop} [decidable c1] {c2: Prop} [decidable c2] (x z: A) : ite c1 (ite c2 x z) z = ite (c1 ∧ c2) x z`
- Description: Provides a supporting lemma about `ite_ite`.

### `filter_distinct_comm`

- Kind: `lemma`
- Source: `src/relational.lean:360`
- Signature: `lemma filter_distinct_comm (p: A → Prop) [decidable_pred p] (i: Z[A]) : filter p (distinct i) = distinct (filter p i)`
- Description: Shows that `filter_distinct` commutes with the compared operation.

### `product_distinct_comm`

- Kind: `theorem`
- Source: `src/relational.lean:368`
- Signature: `theorem product_distinct_comm (i1: Z[A]) (i2: Z[B]) : is_bag i1 → is_bag i2 → product (distinct i1) (distinct i2) = distinct (product i1 i2)`
- Description: Shows that `product_distinct` commutes with the compared operation.

### `join_distinct_comm`

- Kind: `theorem`
- Source: `src/relational.lean:380`
- Signature: `theorem join_distinct_comm (π1: A → C) (π2: B → C) (i1: Z[A]) (i2: Z[B]) : is_bag i1 → is_bag i2 → equi_join π1 π2 (distinct i1) (distinct i2) = distinct (equi_join π1 π2 i1 i2)`
- Description: Shows that `join_distinct` commutes with the compared operation.

### `intersect_distinct_comm`

- Kind: `theorem`
- Source: `src/relational.lean:390`
- Signature: `theorem intersect_distinct_comm (i1 i2: Z[A]) : is_bag i1 → is_bag i2 → distinct i1 ∩ distinct i2 = distinct (i1 ∩ i2)`
- Description: Shows that `intersect_distinct` commutes with the compared operation.

### `map_at_distinct_none`

- Kind: `private lemma`
- Source: `src/relational.lean:401`
- Signature: `private lemma map_at_distinct_none (f: A → B) (i: Z[A]) (b: B) : is_bag i → (∀ a, a ∈ i → f a ≠ b) → flatmap_at (λ a, {f a}) i.distinct b = 0`
- Description: Internal helper: provides a supporting lemma about `map` in the stated setting.

### `map_inj_distinct_comm`

- Kind: `theorem`
- Source: `src/relational.lean:415`
- Signature: `theorem map_inj_distinct_comm (f: A → B) [f_inj: function.injective f] (i: Z[A]) : is_bag i → distinct (zset.map f i) = zset.map f (distinct i)`
- Description: Shows that `map_inj_distinct` commutes with the compared operation.

### `distinct_idem`

- Kind: `lemma`
- Source: `src/relational.lean:452`
- Signature: `lemma distinct_idem (i: Z[A]) : distinct (distinct i) = distinct i`
- Description: Proves idempotence of `distinct`.

### `filter_distinct_dedup`

- Kind: `theorem`
- Source: `src/relational.lean:459`
- Signature: `theorem filter_distinct_dedup (p: A → Prop) [decidable_pred p] (i: Z[A]) : distinct (filter p (distinct i)) = distinct (filter p i)`
- Description: Provides a supporting lemma about `filter` in the stated setting.

### `map_distinct_dedup`

- Kind: `theorem`
- Source: `src/relational.lean:465`
- Signature: `theorem map_distinct_dedup (f: A → B) (i: Z[A]) : is_bag i → distinct (zset.map f (distinct i)) = distinct (zset.map f i)`
- Description: Provides a supporting lemma about `map` in the stated setting.

### `add_distinct_dedup`

- Kind: `theorem`
- Source: `src/relational.lean:486`
- Signature: `theorem add_distinct_dedup (i1 i2: Z[A]) : is_bag i1 → is_bag i2 → distinct (i1.distinct + i2.distinct) = distinct (i1 + i2)`
- Description: Provides a supporting theorem about `add_distinct_dedup`.

### `product_distinct_dedup`

- Kind: `theorem`
- Source: `src/relational.lean:496`
- Signature: `theorem product_distinct_dedup (i1: Z[A]) (i2: Z[B]) : is_bag i1 → is_bag i2 → distinct (product (distinct i1) (distinct i2)) = distinct (product i1 i2)`
- Description: Provides a supporting lemma about `product` in the stated setting.

### `join_distinct_dedup`

- Kind: `theorem`
- Source: `src/relational.lean:505`
- Signature: `theorem join_distinct_dedup (π1: A → C) (π2: B → C) (i1: Z[A]) (i2: Z[B]) : is_bag i1 → is_bag i2 → distinct (equi_join π1 π2 (distinct i1) (distinct i2)) = distinct (equi_join π1 π2 i1 i2)`
- Description: Provides a supporting lemma about equi-join in the stated setting.

### `intersect_distinct_dedup`

- Kind: `theorem`
- Source: `src/relational.lean:514`
- Signature: `theorem intersect_distinct_dedup (i1 i2: Z[A]) : is_bag i1 → is_bag i2 → distinct (distinct i1 ∩ distinct i2) = distinct (i1 ∩ i2)`
- Description: Provides a supporting lemma about `intersect` in the stated setting.

## `relational_example.lean`

A worked relational query example that derives optimized batch and incremental plans from the DBSP algebra.

Declarations in this module: **22**.

### `πxy`

- Kind: `def`
- Source: `src/relational_example.lean:20`
- Signature: `def πxy (S: schema) (t : (S.T1X × S.Id) × (S.Id × S.T2Y)) : S.T1X × S.T2Y`
- Description: Projects a joined tuple down to the `(x, y)` payload kept by the example query.

### `t1`

- Kind: `def`
- Source: `src/relational_example.lean:39`
- Signature: `def t1 : Z[S.T] → Z[S.T1X × S.Id]`
- Description: Defines the first optimized subquery over table `T`.

### `t2`

- Kind: `def`
- Source: `src/relational_example.lean:42`
- Signature: `def t2 : Z[S.R] → Z[S.Id × S.T2Y]`
- Description: Defines the second optimized subquery over table `R`.

### `V`

- Kind: `def`
- Source: `src/relational_example.lean:45`
- Signature: `def V : Z[S.T] → Z[S.R] → Z[S.T1X × S.T2Y]`
- Description: Defines the original relational query from the example schema.

### `pos_equiv`

- Kind: `def`
- Source: `src/relational_example.lean:53`
- Signature: `def pos_equiv {A B: Type} [decidable_eq A] [decidable_eq B] (f1 f2: Z[A] → Z[B])`
- Description: Defines equivalence of unary queries on positive inputs only.

### `pos_equiv2`

- Kind: `def`
- Source: `src/relational_example.lean:57`
- Signature: `def pos_equiv2 {A B C: Type} [decidable_eq A] [decidable_eq B] [decidable_eq C] (f1 f2: Z[A] → Z[B] → Z[C])`
- Description: Defines equivalence of binary queries on positive inputs only.

### `same`

- Kind: `def`
- Source: `src/relational_example.lean:75`
- Signature: `def same {A : Type} (x y: A)`
- Description: Defines an irreducible alias for equality that is used to guide automation.

### `same_def`

- Kind: `lemma`
- Source: `src/relational_example.lean:76`
- Signature: `lemma same_def {A} (x y: A) : same x y = (x = y)`
- Description: Unfolds the `same` wrapper back to ordinary equality.

### `same_intro`

- Kind: `lemma`
- Source: `src/relational_example.lean:79`
- Signature: `lemma same_intro {A: Type} (x y: A) : same x y → x = y`
- Description: Turns a proof of `same` into an ordinary equality.

### `same_elim`

- Kind: `lemma`
- Source: `src/relational_example.lean:82`
- Signature: `lemma same_elim {A: Type} (x: A) : same x x`
- Description: Builds a trivial `same` proof from reflexive equality.

### `t1_opt_goal`

- Kind: `def`
- Source: `src/relational_example.lean:91`
- Signature: `def t1_opt_goal : sig (Z[S.T] → Z[S.T1X × S.Id]) (λ opt, t1 S =≤= opt)`
- Description: Packages a candidate optimization for `t1` together with its positive-input correctness proof.

### `t1_opt`

- Kind: `def`
- Source: `src/relational_example.lean:105`
- Signature: `def t1_opt`
- Description: Extracts the optimized `t1` implementation from `t1_opt_goal`.

### `t1_opt_ok`

- Kind: `def`
- Source: `src/relational_example.lean:106`
- Signature: `def t1_opt_ok : t1 S =≤= t1_opt S`
- Description: Extracts the proof that `t1_opt` is positively equivalent to `t1`.

### `t2_opt_goal`

- Kind: `def`
- Source: `src/relational_example.lean:108`
- Signature: `def t2_opt_goal : sig (Z[S.R] → Z[S.Id × S.T2Y]) (λ opt, t2 S =≤= opt)`
- Description: Packages a candidate optimization for `t2` together with its proof.

### `v_opt_goal`

- Kind: `def`
- Source: `src/relational_example.lean:121`
- Signature: `def v_opt_goal : sig (Z[S.T] → Z[S.R] → Z[S.T1X × S.T2Y]) (λ opt, V S =≤2= opt)`
- Description: Packages a candidate optimization for the full binary query `V`.

### `Vopt`

- Kind: `def`
- Source: `src/relational_example.lean:139`
- Signature: `def Vopt (t1: Z[S.T]) (t2: Z[S.R])`
- Description: Defines the paper-style optimized Z-set query.

### `v_opt_ok`

- Kind: `lemma`
- Source: `src/relational_example.lean:146`
- Signature: `lemma v_opt_ok : V S =≤2= Vopt S`
- Description: Shows that the optimized query `Vopt` is positively equivalent to the original query `V`.

### `v_lifted`

- Kind: `lemma`
- Source: `src/relational_example.lean:149`
- Signature: `lemma v_lifted : ↑²(Vopt S) = λ t1 t2, ↑↑distinct (↑↑(zset.map (πxy S)) (↑²(equi_join prod.snd prod.fst) (↑↑(zset.map schema.π1) (↑↑(filter schema.σT) t1)) (↑↑(zset.map schema.π2) (↑↑(filter schema.σR) t2))))`
- Description: Expands the lifted form of `Vopt` into the operator-level expression used for incrementalization.

### `VΔ1`

- Kind: `def`
- Source: `src/relational_example.lean:158`
- Signature: `def VΔ1 (t1: stream Z[S.T]) (t2: stream Z[S.R])`
- Description: Defines an intermediate incremental circuit for `Vopt`.

### `VΔ1_ok`

- Kind: `lemma`
- Source: `src/relational_example.lean:163`
- Signature: `lemma VΔ1_ok : ↑²(Vopt S)^Δ2 = VΔ1 S`
- Description: Shows that the first incremental plan `VΔ1` matches the abstract incrementalization of `Vopt`.

### `VΔ`

- Kind: `def`
- Source: `src/relational_example.lean:181`
- Signature: `def VΔ (t1: stream Z[S.T]) (t2: stream Z[S.R])`
- Description: Defines the fully simplified incremental query used in the worked example.

### `VΔ_ok`

- Kind: `theorem`
- Source: `src/relational_example.lean:186`
- Signature: `theorem VΔ_ok : ↑²(Vopt S)^Δ2 = VΔ S`
- Description: Shows that the final simplified incremental plan `VΔ` still matches the incrementalization of `Vopt`.

## `relational_incremental.lean`

Incremental rules for relational operators, especially `distinct`, `map`, `flatmap`, joins, and filters lifted to streams.

Declarations in this module: **11**.

### `distinct_H_at`

- Kind: `def`
- Source: `src/relational_incremental.lean:12`
- Signature: `def distinct_H_at (i d: Z[A]) : A → ℤ`
- Description: Defines the pointwise helper used to incrementalize `distinct`.

### `distinct_H`

- Kind: `def`
- Source: `src/relational_incremental.lean:18`
- Signature: `def distinct_H (i d: Z[A]) : Z[A]`
- Description: Packages `distinct_H_at` as a Z-set operator.

### `distinct_H_apply`

- Kind: `lemma`
- Source: `src/relational_incremental.lean:23`
- Signature: `lemma distinct_H_apply (i d: Z[A]) (x: A) : distinct_H i d x = distinct_H_at i d x`
- Description: Gives the pointwise evaluation rule for `distinct_H`.

### `distinct_incremental`

- Kind: `def`
- Source: `src/relational_incremental.lean:33`
- Signature: `def distinct_incremental : stream Z[A] → stream Z[A]`
- Description: Defines the incremental stream operator for relational `distinct`.

### `distinct_incremental_ok`

- Kind: `theorem`
- Source: `src/relational_incremental.lean:36`
- Signature: `theorem distinct_incremental_ok : (↑↑ distinct)^Δ = @distinct_incremental A _`
- Description: Shows that `distinct_incremental` matches its intended semantics.

### `flatmap_incremental`

- Kind: `lemma`
- Source: `src/relational_incremental.lean:60`
- Signature: `lemma flatmap_incremental (f: A → Z[B]) : ↑↑(zset.flatmap f)^Δ = ↑↑(zset.flatmap f)`
- Description: Provides a supporting lemma about `flatmap` in the stated setting.

### `map_incremental`

- Kind: `lemma`
- Source: `src/relational_incremental.lean:69`
- Signature: `lemma map_incremental (f: A → B) : ↑↑(zset.map f)^Δ = ↑↑(zset.map f)`
- Description: Provides a supporting lemma about `map` in the stated setting.

### `lifting_map_incremental`

- Kind: `lemma`
- Source: `src/relational_incremental.lean:74`
- Signature: `lemma lifting_map_incremental {A B} [decidable_eq A] [decidable_eq B] (f: A → B) : ↑↑(↑↑(zset.map f))^Δ = ↑↑(↑↑(zset.map f))`
- Description: Provides a supporting lemma about lifted operators.

### `map_incremental_unfolded`

- Kind: `lemma`
- Source: `src/relational_incremental.lean:84`
- Signature: `lemma map_incremental_unfolded (f: A → B) (s: stream Z[A]) : D (↑↑(zset.map f) (I s)) = ↑↑(zset.map f) s`
- Description: Spells out the fully unfolded form of `map_incremental`.

### `equi_join_incremental`

- Kind: `theorem`
- Source: `src/relational_incremental.lean:93`
- Signature: `theorem equi_join_incremental (π1: A → C) (π2: B → C) : ↑²(equi_join π1 π2)^Δ2 = times_incremental ↑²(equi_join π1 π2)`
- Description: Provides a supporting lemma about equi-join in the stated setting.

### `filter_incremental`

- Kind: `lemma`
- Source: `src/relational_incremental.lean:106`
- Signature: `lemma filter_incremental (p: A → Prop) [decidable_pred p] : ↑↑(filter p)^Δ = ↑↑(filter p)`
- Description: Provides a supporting lemma about `filter` in the stated setting.

## `stream.lean`

Foundational stream definitions, agreement relations, and prefix truncation (`cut`).

Declarations in this module: **20**.

### `stream`

- Kind: `def`
- Source: `src/stream.lean:29`
- Signature: `def stream (a: Type u) : Type u`
- Description: Defines a stream as a function from natural-number time to values.

### `agree_upto`

- Kind: `def`
- Source: `src/stream.lean:35`
- Signature: `def agree_upto (n: ℕ) (s₁ s₂: stream a)`
- Description: Defines when two streams agree up to a finite time horizon.

### `stream_le_ext`

- Kind: `lemma`
- Source: `src/stream.lean:43`
- Signature: `lemma stream_le_ext [partial_order a] (s1 s2: stream a) : s1 ≤ s2 = (∀ t, s1 t ≤ s2 t)`
- Description: Provides a supporting lemma about `stream_le_ext`.

### `agree_refl`

- Kind: `lemma`
- Source: `src/stream.lean:49`
- Signature: `lemma agree_refl (n: ℕ) : ∀ (s: stream a), s ==n== s`
- Description: Proves reflexivity of `agree`.

### `agree_symm`

- Kind: `lemma`
- Source: `src/stream.lean:57`
- Signature: `lemma agree_symm (n: ℕ) : ∀ (s1 s2: stream a), s1 ==n== s2 → s2 ==n== s1`
- Description: Proves symmetry of `agree`.

### `agree_trans`

- Kind: `lemma`
- Source: `src/stream.lean:65`
- Signature: `lemma agree_trans {n: ℕ} : ∀ (s1 s2 s3: stream a), s1 ==n== s2 → s2 ==n== s3 → s1 ==n== s3`
- Description: Proves transitivity of `agree`.

### `agree_everywhere_eq`

- Kind: `theorem`
- Source: `src/stream.lean:79`
- Signature: `theorem agree_everywhere_eq (s s': stream a) : s = s' ↔ (∀ n, s ==n== s')`
- Description: Shows that agreement everywhere implies actual equality for `agree`.

### `agree_upto_weaken`

- Kind: `lemma`
- Source: `src/stream.lean:91`
- Signature: `lemma agree_upto_weaken {s s': stream a} (n n': ℕ) : s ==n== s' → n' ≤ n → s ==n'== s'`
- Description: Weakens the horizon or hypothesis used in `agree_upto`.

### `agree_upto_weaken1`

- Kind: `lemma`
- Source: `src/stream.lean:101`
- Signature: `lemma agree_upto_weaken1 {s s': stream a} (n: ℕ) : s ==n.succ== s' → s ==n== s'`
- Description: Specializes the weakening rule for `agree_upto` by one time step.

### `agree_upto_0`

- Kind: `lemma`
- Source: `src/stream.lean:109`
- Signature: `lemma agree_upto_0 (s s': stream a) : s ==0== s' ↔ s 0 = s' 0`
- Description: Shows the zero-case behavior of `agree_upto`.

### `agree_upto_extend`

- Kind: `lemma`
- Source: `src/stream.lean:122`
- Signature: `lemma agree_upto_extend (n: nat) (s s': stream a) : s ==n== s' → s n.succ = s' n.succ → s ==n.succ== s'`
- Description: Extends `agree_upto` by one additional time step.

### `cut`

- Kind: `def`
- Source: `src/stream.lean:140`
- Signature: `def cut (s: stream a) (t: ℕ) : stream a`
- Description: Defines prefix truncation of a stream at a chosen time.

### `cut_at_0`

- Kind: `lemma`
- Source: `src/stream.lean:143`
- Signature: `lemma cut_at_0 (s: stream a) : cut s 0 = 0`
- Description: Shows the zero-case behavior of `cut_at`.

### `cut_0`

- Kind: `lemma`
- Source: `src/stream.lean:150`
- Signature: `lemma cut_0 : cut (0 : stream a) = 0`
- Description: Shows the zero-case behavior of `cut`.

### `cut_cut`

- Kind: `theorem`
- Source: `src/stream.lean:156`
- Signature: `theorem cut_cut (s: stream a) (t1 t2: ℕ) : cut (cut s t1) t2 = cut s (min t1 t2)`
- Description: Provides a supporting theorem about `cut_cut`.

### `cut_comm`

- Kind: `theorem`
- Source: `src/stream.lean:166`
- Signature: `theorem cut_comm (s: stream a) (t1 t2: ℕ) : cut (cut s t1) t2 = cut (cut s t2) t1`
- Description: Shows that `cut` commutes with the compared operation.

### `cut_idem`

- Kind: `theorem`
- Source: `src/stream.lean:173`
- Signature: `theorem cut_idem (s: stream a) (t: ℕ) : cut (cut s t) t = cut s t`
- Description: Proves idempotence of `cut`.

### `agree_upto_cut`

- Kind: `theorem`
- Source: `src/stream.lean:180`
- Signature: `theorem agree_upto_cut (s1 s2: stream a) (n: ℕ) : s1 ==n== s2 ↔ cut s1 n.succ = cut s2 n.succ`
- Description: Provides a supporting lemma about finite-horizon stream agreement.

### `cut_agree_succ`

- Kind: `lemma`
- Source: `src/stream.lean:197`
- Signature: `lemma cut_agree_succ (s1 s2: stream a) (t: ℕ) : cut s1 t = cut s2 t → s1 t = s2 t → cut s1 t.succ = cut s2 t.succ`
- Description: Gives the successor-step form of `cut_agree`.

### `agree_with_cut`

- Kind: `theorem`
- Source: `src/stream.lean:213`
- Signature: `theorem agree_with_cut (s: stream a) (n: ℕ) : s ==n== cut s n.succ`
- Description: Provides a supporting theorem about `agree_with_cut`.

## `stream_elim.lean`

Stream introduction/elimination on eventually-zero streams, bridging scalar values and finite-support streams.

Declarations in this module: **28**.

### `δ0`

- Kind: `def`
- Source: `src/stream_elim.lean:13`
- Signature: `def δ0 (x:a) : stream a`
- Description: Injects a scalar value into a stream with all mass at time `0`.

### `δ0_apply`

- Kind: `lemma`
- Source: `src/stream_elim.lean:17`
- Signature: `lemma δ0_apply (x: a) (n: ℕ) : δ0 x n = if n = 0 then x else 0`
- Description: Gives the pointwise evaluation rule for `δ0`.

### `δ0_0`

- Kind: `lemma`
- Source: `src/stream_elim.lean:22`
- Signature: `lemma δ0_0 : δ0 (0: a) = 0`
- Description: Shows the zero-case behavior of `δ0`.

### `zero_after`

- Kind: `def`
- Source: `src/stream_elim.lean:27`
- Signature: `def zero_after (s: stream a) (n: ℕ)`
- Description: Defines the predicate that a stream is zero after some cutoff time.

### `zero_after_ge`

- Kind: `lemma`
- Source: `src/stream_elim.lean:29`
- Signature: `lemma zero_after_ge {s: stream a} {n1: ℕ} (pf1: zero_after s n1) : ∀ n2 ≥ n1, zero_after s n2`
- Description: Records the greater-or-equal monotonicity fact for `zero_after`.

### `δ0_zero_after`

- Kind: `def`
- Source: `src/stream_elim.lean:37`
- Signature: `def δ0_zero_after (x:a) : zero_after (δ0 x) 1`
- Description: Specializes `zero_after` to the stream created by `δ0`.

### `drop`

- Kind: `def`
- Source: `src/stream_elim.lean:45`
- Signature: `def drop (k: ℕ) (s: stream a) : stream a`
- Description: Drops the first `n` timestamps of a stream.

### `sum_vals_split`

- Kind: `lemma`
- Source: `src/stream_elim.lean:48`
- Signature: `lemma sum_vals_split (s: stream a) (n k: ℕ) : sum_vals s (n + k) = sum_vals s n + sum_vals (drop n s) k`
- Description: Provides a supporting summation lemma about `sum_vals_split`.

### `sum_vals_zero_ge`

- Kind: `lemma`
- Source: `src/stream_elim.lean:60`
- Signature: `lemma sum_vals_zero_ge (s: stream a) (n m:ℕ) (hz: zero_after s n) (hge: m ≥ n) : sum_vals s n = sum_vals s m`
- Description: Records the greater-or-equal monotonicity fact for `sum_vals_zero`.

### `sum_vals_eq_helper`

- Kind: `lemma`
- Source: `src/stream_elim.lean:70`
- Signature: `lemma sum_vals_eq_helper (s: stream a) (n1 n2: ℕ) (hz1: zero_after s n1) (hz2: zero_after s n2) : n1 ≤ n2 → sum_vals s n1 = sum_vals s n2`
- Description: Auxiliary equality lemma used to reason about prefix sums when eliminating streams.

### `sum_vals_eq`

- Kind: `lemma`
- Source: `src/stream_elim.lean:80`
- Signature: `lemma sum_vals_eq (s: stream a) (n1 n2: ℕ) (hz1: zero_after s n1) (hz2: zero_after s n2) : sum_vals s n1 = sum_vals s n2`
- Description: Characterizes `sum_vals` in terms of the corresponding `drop`ped stream.

### `stream_elim`

- Kind: `noncomputable def`
- Source: `src/stream_elim.lean:88`
- Signature: `noncomputable def stream_elim (s: stream a) : a`
- Description: Eliminates an eventually-zero stream back to a single accumulated value.

### `stream_elim_zero_after`

- Kind: `lemma`
- Source: `src/stream_elim.lean:94`
- Signature: `lemma stream_elim_zero_after (s: stream a) (n:ℕ) (pf:zero_after s n) : stream_elim s = sum_vals s n`
- Description: Shows that `stream_elim` equals a finite prefix sum once the stream is known to be zero beyond that prefix.

### `stream_elim_0`

- Kind: `lemma`
- Source: `src/stream_elim.lean:112`
- Signature: `lemma stream_elim_0 : ∫ (0: stream a) = 0`
- Description: Shows the zero-case behavior of `stream_elim`.

### `stream_elim_delta`

- Kind: `theorem`
- Source: `src/stream_elim.lean:119`
- Signature: `theorem stream_elim_delta (x: a) : ∫ (δ0 x) = x`
- Description: Provides a supporting lemma about stream elimination.

### `delta_linear`

- Kind: `theorem`
- Source: `src/stream_elim.lean:127`
- Signature: `theorem delta_linear : ∀ (x y: a), δ0 (x + y) = δ0 x + δ0 y`
- Description: Shows that `delta` is linear.

### `delta_incremental`

- Kind: `lemma`
- Source: `src/stream_elim.lean:136`
- Signature: `lemma delta_incremental : ↑↑(@δ0 a _)^Δ = ↑↑δ0`
- Description: Shows that stream introduction `δ0` is already incrementalized.

### `sum_vals_linear`

- Kind: `lemma`
- Source: `src/stream_elim.lean:144`
- Signature: `lemma sum_vals_linear (s1 s2: stream a) (n: ℕ) : sum_vals (s1 + s2) n = sum_vals s1 n + sum_vals s2 n`
- Description: Shows that `sum_vals` is linear.

### `sum_zero_after`

- Kind: `lemma`
- Source: `src/stream_elim.lean:151`
- Signature: `lemma sum_zero_after {s1 s2: stream a} {n1: ℕ} (pf1: zero_after s1 n1) {n2: ℕ} (pf2: zero_after s2 n2) : zero_after (s1 + s2) (if n1 ≥ n2 then n1 else n2)`
- Description: Provides a supporting summation lemma about `sum_zero_after`.

### `sub_zero_after`

- Kind: `lemma`
- Source: `src/stream_elim.lean:168`
- Signature: `lemma sub_zero_after {s1 s2: stream a} {n1: ℕ} (pf1: zero_after s1 n1) {n2: ℕ} (pf2: zero_after s2 n2) : zero_after (s1 - s2) (if n1 ≥ n2 then n1 else n2)`
- Description: Provides a supporting lemma about `sub_zero_after`.

### `stream_elim_linear`

- Kind: `theorem`
- Source: `src/stream_elim.lean:185`
- Signature: `theorem stream_elim_linear (s1 s2: stream a) (n1: ℕ) (pf1: zero_after s1 n1) (n2: ℕ) (pf2: zero_after s2 n2) : ∫ (s1 + s2) = ∫ s1 + ∫ s2`
- Description: Shows that `stream_elim` is linear on eventually-zero streams.

### `stream_elim_time_invariant`

- Kind: `lemma`
- Source: `src/stream_elim.lean:204`
- Signature: `lemma stream_elim_time_invariant : time_invariant ↑↑(@stream_elim a _)`
- Description: Shows that `stream_elim` is time invariant on its nested-stream formulation.

### `integral_zero`

- Kind: `lemma`
- Source: `src/stream_elim.lean:210`
- Signature: `lemma integral_zero (s: stream a) (n: ℕ) : zero_after (I s) n → zero_after s n.succ`
- Description: Shows that integrating a zero-after stream yields zero at the boundary used by elimination.

### `integral_nested_unfold`

- Kind: `lemma`
- Source: `src/stream_elim.lean:226`
- Signature: `lemma integral_nested_unfold (s: stream (stream a)) (t: ℕ) : 0 < t → I s t = s t + I s (t-1)`
- Description: Unfolds nested integration in the form needed for stream elimination proofs.

### `integral_zero'`

- Kind: `lemma`
- Source: `src/stream_elim.lean:238`
- Signature: `lemma integral_zero' (s: stream (stream a)) (t n: ℕ) : zero_after (I s t) n → zero_after (I s (t-1)) n → zero_after (s t) n.succ`
- Description: A second zero-after integral lemma specialized to the nested setting.

### `integral_delta`

- Kind: `lemma`
- Source: `src/stream_elim.lean:282`
- Signature: `lemma integral_delta (x:a) : I (δ0 x) = λ _n, x`
- Description: Shows that integrating a `δ0` stream recovers a constant stream.

### `integral_delta_apply`

- Kind: `lemma`
- Source: `src/stream_elim.lean:292`
- Signature: `lemma integral_delta_apply (x:a) (n:ℕ) : I (δ0 x) n = x`
- Description: Gives the pointwise form of `integral_delta`.

### `nested_zpp`

- Kind: `lemma`
- Source: `src/stream_elim.lean:300`
- Signature: `lemma nested_zpp (Q: operator a b) : time_invariant Q → ∫ (Q (δ0 0)) = 0`
- Description: Shows that the nested integral operator preserves zero at time zero.

## `zset.lean`

Foundations of Z-sets as finitely supported integer-valued functions, including set/bag views and core combinators like `distinct`, `flatmap`, and `map`.

Declarations in this module: **58**.

### `zset`

- Kind: `def`
- Source: `src/zset.lean:48`
- Signature: `def zset (A: Type)`
- Description: Defines a Z-set as a finitely supported function from elements to integer multiplicities.

### `graph`

- Kind: `def`
- Source: `src/zset.lean:63`
- Signature: `def graph (m: Z[A]) : multiset (A × ℤ)`
- Description: Views a Z-set as a multiset of `(element, weight)` pairs.

### `graph_list`

- Kind: `def`
- Source: `src/zset.lean:66`
- Signature: `def graph_list [linear_order A] (m: Z[A]) : list (A ×ₗ ℤ)`
- Description: Lists the graph representation of a Z-set in sorted order.

### `elements`

- Kind: `def`
- Source: `src/zset.lean:69`
- Signature: `def elements [linear_order A] (m: Z[A]) : list A`
- Description: Lists the elements that appear in a Z-set support.

### `add_apply`

- Kind: `lemma`
- Source: `src/zset.lean:81`
- Signature: `lemma add_apply (m1 m2: Z[A]) (a: A) : (m1 + m2) a = m1 a + m2 a`
- Description: Gives the pointwise evaluation rule for `add`.

### `sub_apply`

- Kind: `lemma`
- Source: `src/zset.lean:84`
- Signature: `lemma sub_apply (m1 m2: Z[A]) (a: A) : (m1 - m2) a = m1 a - m2 a`
- Description: Gives the pointwise evaluation rule for `sub`.

### `neg_apply`

- Kind: `lemma`
- Source: `src/zset.lean:87`
- Signature: `lemma neg_apply (m: Z[A]) (a: A) : (-m) a = -(m a)`
- Description: Gives the pointwise evaluation rule for `neg`.

### `add_support`

- Kind: `lemma`
- Source: `src/zset.lean:89`
- Signature: `lemma add_support (m1 m2: Z[A]) : (m1 + m2).support = (m1.support ∪ m2.support).filter (λ a, m1 a + m2 a ≠ 0)`
- Description: Characterizes the support of `add`.

### `empty`

- Kind: `protected def`
- Source: `src/zset.lean:100`
- Signature: `protected def empty : Z[A]`
- Description: Defines the empty Z-set.

### `single`

- Kind: `protected def`
- Source: `src/zset.lean:103`
- Signature: `protected def single (a:A) : Z[A]`
- Description: Defines the singleton Z-set containing one copy of a value.

### `insert`

- Kind: `protected def`
- Source: `src/zset.lean:107`
- Signature: `protected def insert (a:A) (m: Z[A])  : Z[A]`
- Description: Defines insertion as addition with a singleton Z-set.

### `emptyc_apply`

- Kind: `lemma`
- Source: `src/zset.lean:111`
- Signature: `lemma emptyc_apply (a:A) : (∅ : Z[A]) a = 0`
- Description: Gives the pointwise evaluation rule for `emptyc`.

### `single_apply`

- Kind: `lemma`
- Source: `src/zset.lean:114`
- Signature: `lemma single_apply (a:A) (a': A) : ({a} : Z[A]) a' = if a = a' then 1 else 0`
- Description: Gives the pointwise evaluation rule for `single`.

### `insert_apply`

- Kind: `lemma`
- Source: `src/zset.lean:122`
- Signature: `lemma insert_apply (a: A) (m: Z[A]) (a': A) : has_insert.insert a m a' = m a' + if a = a' then 1 else 0`
- Description: Gives the pointwise evaluation rule for `insert`.

### `zset_le_ext`

- Kind: `lemma`
- Source: `src/zset.lean:159`
- Signature: `lemma zset_le_ext (m1 m2: Z[A]) : m1 ≤ m2 = (∀ a, m1 a ≤ m2 a)`
- Description: Provides a supporting lemma about `zset_le_ext`.

### `elem_eq`

- Kind: `lemma`
- Source: `src/zset.lean:162`
- Signature: `lemma elem_eq (a: A) (m: zset A) : (a ∈ m) = (a ∈ m.support)`
- Description: States an equality characterizing `elem`.

### `from_set`

- Kind: `protected def`
- Source: `src/zset.lean:164`
- Signature: `protected def from_set (s: finset A) : Z[A]`
- Description: Encodes an ordinary finite set as a Z-set with multiplicity `1` everywhere in the set.

### `from_set_0`

- Kind: `lemma`
- Source: `src/zset.lean:167`
- Signature: `lemma from_set_0 : zset.from_set (∅ : finset A) = dfinsupp.mk ∅ (λ a, 1)`
- Description: Shows the zero-case behavior of `from_set`.

### `from_set_apply`

- Kind: `lemma`
- Source: `src/zset.lean:171`
- Signature: `lemma from_set_apply (s: finset A) (a: A) : zset.from_set s a = if (a ∈ s) then 1 else 0`
- Description: Gives the pointwise evaluation rule for `from_set`.

### `from_set_support`

- Kind: `lemma`
- Source: `src/zset.lean:176`
- Signature: `lemma from_set_support (s: finset A) : (zset.from_set s).support = s`
- Description: Characterizes the support of `from_set`.

### `to_set`

- Kind: `protected def`
- Source: `src/zset.lean:179`
- Signature: `protected def to_set (s: Z[A]) : finset A`
- Description: Forgets multiplicities and extracts the support of a Z-set as a finite set.

### `elem_to_set`

- Kind: `lemma`
- Source: `src/zset.lean:182`
- Signature: `lemma elem_to_set (a: A) (m: Z[A]) : a ∈ m.to_set ↔ a ∈ m`
- Description: Provides a supporting lemma about `elem_to_set`.

### `elem_from_set`

- Kind: `lemma`
- Source: `src/zset.lean:185`
- Signature: `lemma elem_from_set (a: A) (s: finset A) : a ∈ zset.from_set s ↔ a ∈ s`
- Description: Provides a supporting lemma about `elem_from_set`.

### `to_from_set`

- Kind: `lemma`
- Source: `src/zset.lean:190`
- Signature: `lemma to_from_set (s: finset A) [∀ a, decidable (a ∈ s)] : (zset.from_set s).to_set = s`
- Description: Provides a supporting lemma about `to_from_set`.

### `is_set`

- Kind: `def`
- Source: `src/zset.lean:201`
- Signature: `def is_set (m: Z[A])`
- Description: Defines the predicate that a Z-set is set-valued, with only multiplicity `1` on supported elements.

### `is_bag`

- Kind: `def`
- Source: `src/zset.lean:202`
- Signature: `def is_bag (m: Z[A])`
- Description: Defines the predicate that a Z-set is bag-valued, with all multiplicities nonnegative.

### `fun_positive`

- Kind: `def`
- Source: `src/zset.lean:203`
- Signature: `def fun_positive (f: Z[A] → Z[B])`
- Description: Defines positivity preservation for unary functions on Z-sets.

### `fun_positive2`

- Kind: `def`
- Source: `src/zset.lean:204`
- Signature: `def fun_positive2 (f: Z[A] → Z[B] → Z[C])`
- Description: Defines positivity preservation for binary functions on Z-sets.

### `elem_mp`

- Kind: `lemma`
- Source: `src/zset.lean:208`
- Signature: `lemma elem_mp (m: Z[A]) (a: A) : a ∈ m ↔ m a ≠ 0`
- Description: Provides a supporting lemma about `elem_mp`.

### `not_elem_mp`

- Kind: `lemma`
- Source: `src/zset.lean:210`
- Signature: `lemma not_elem_mp (a: A) (m: Z[A]) : a ∉ m ↔ m a = 0`
- Description: Provides a supporting lemma about `not_elem_mp`.

### `is_set_or`

- Kind: `lemma`
- Source: `src/zset.lean:213`
- Signature: `lemma is_set_or (s: Z[A]) : is_set s ↔ ∀ a, s a = 0 ∨ s a = 1`
- Description: Provides a supporting lemma about `is_set_or`.

### `is_set_0`

- Kind: `lemma`
- Source: `src/zset.lean:225`
- Signature: `lemma is_set_0 : is_set (0: Z[A])`
- Description: Shows the zero-case behavior of `is_set`.

### `set_is_bag`

- Kind: `lemma`
- Source: `src/zset.lean:230`
- Signature: `lemma set_is_bag (s: Z[A]) : is_set s -> is_bag s`
- Description: Provides a supporting lemma about `set_is_bag`.

### `elem_single`

- Kind: `lemma`
- Source: `src/zset.lean:239`
- Signature: `lemma elem_single (a x:A) : x ∈ ({a} : Z[A]) ↔ a = x`
- Description: Provides a supporting lemma about `elem_single`.

### `support_single`

- Kind: `lemma`
- Source: `src/zset.lean:246`
- Signature: `lemma support_single (a: A) : ({a} : Z[A]).support = {a}`
- Description: Provides a supporting lemma about `support_single`.

### `distinct`

- Kind: `def`
- Source: `src/zset.lean:252`
- Signature: `def distinct (m: Z[A]) : Z[A]`
- Description: Defines `distinct`, which clamps every positive multiplicity to `1` and removes nonpositive entries.

### `distinct_apply`

- Kind: `lemma`
- Source: `src/zset.lean:256`
- Signature: `lemma distinct_apply (m: Z[A]) (a: A) : distinct m a = if m a > 0 then 1 else 0`
- Description: Gives the pointwise evaluation rule for `distinct`.

### `union_disjoint_l`

- Kind: `lemma`
- Source: `src/zset.lean:267`
- Signature: `lemma union_disjoint_l (s1 s2: finset A) : s1 ∪ s2 = s1.disj_union (s2 \ s1) finset.disjoint_sdiff`
- Description: Provides a supporting lemma about `union_disjoint_l`.

### `filter_filter_comm`

- Kind: `lemma`
- Source: `src/zset.lean:273`
- Signature: `lemma filter_filter_comm (p q : A → Prop) [decidable_pred p] [decidable_pred q] (s: finset A) : finset.filter p (finset.filter q s) = finset.filter q (finset.filter p s)`
- Description: Shows that `filter_filter` commutes with the compared operation.

### `sum_union_zero_l`

- Kind: `lemma`
- Source: `src/zset.lean:281`
- Signature: `lemma sum_union_zero_l {α: Type} [add_comm_monoid α] (f: A → α) (s s': finset A) : (∀ x, x ∈ s' → x ∉ s → f x = 0) → finset.sum (s ∪ s') f = finset.sum s f`
- Description: Proves a sum-over-union identity when the function vanishes on the left-only portion of the union.

### `general_sum_linear`

- Kind: `theorem`
- Source: `src/zset.lean:297`
- Signature: `theorem general_sum_linear (m1 m2: Z[A]) : (∀ a, f a 0 = 0) → (∀ a m1 m2, f a (m1 + m2) = f a m1 + f a m2) → (m1 + m2).support.sum (λ a, f a ((m1 + m2) a)) = m1.support.sum (λ a, f a (m1 a)) + m2.support.sum (λ a, f a (m2 a))`
- Description: A general linearity theorem for support-based sums over Z-sets.

### `flatmap_at`

- Kind: `def`
- Source: `src/zset.lean:340`
- Signature: `def flatmap_at (m: Z[A]) : B → ℤ`
- Description: Defines the pointwise contribution of `flatmap` at a chosen output element.

### `flatmap`

- Kind: `def`
- Source: `src/zset.lean:343`
- Signature: `def flatmap (m: Z[A]) : Z[B]`
- Description: Defines multiset-style `flatmap` for Z-sets.

### `flatmap_apply`

- Kind: `lemma`
- Source: `src/zset.lean:350`
- Signature: `lemma flatmap_apply (m: Z[A]) : ∀ b, m.flatmap f b = flatmap_at f m b`
- Description: Gives the pointwise evaluation rule for `flatmap`.

### `flatmap_0`

- Kind: `theorem`
- Source: `src/zset.lean:362`
- Signature: `theorem flatmap_0 : zset.flatmap f 0 = 0`
- Description: Shows the zero-case behavior of `flatmap`.

### `ite_cases`

- Kind: `private lemma`
- Source: `src/zset.lean:364`
- Signature: `private lemma ite_cases {c: Prop} [decidable c] (x z: A) (p: A → Prop) : p z → p x → p (ite c x z)`
- Description: Internal helper: provides a supporting lemma about `ite_cases`.

### `flatmap_linear`

- Kind: `theorem`
- Source: `src/zset.lean:372`
- Signature: `theorem flatmap_linear (m1 m2: Z[A]) : zset.flatmap f (m1 + m2) = zset.flatmap f m1 + zset.flatmap f m2`
- Description: Shows that `flatmap` is linear.

### `flatmap_from_set_card`

- Kind: `lemma`
- Source: `src/zset.lean:382`
- Signature: `lemma flatmap_from_set_card (s: finset A) (b:B) : (∀ a, (f a).is_set) → zset.flatmap_at f (zset.from_set s) b = finset.card (s.filter (λ (x : A), b ∈ f x))`
- Description: Computes the multiplicity contributed by `flatmap` on a set-encoded input.

### `map_from_set_card`

- Kind: `lemma`
- Source: `src/zset.lean:398`
- Signature: `lemma map_from_set_card (f: A → B) (s: finset A) (b:B) : zset.flatmap_at (λ a, {f a}) (zset.from_set s) b = finset.card (s.filter (λ (x : A), f x = b))`
- Description: Computes the multiplicity of a mapped set-encoded input.

### `map`

- Kind: `protected def`
- Source: `src/zset.lean:413`
- Signature: `protected def map (m: Z[A]) : Z[B]`
- Description: Defines `map` as a special case of `flatmap` to singleton outputs.

### `flatmap_map_at`

- Kind: `lemma`
- Source: `src/zset.lean:415`
- Signature: `lemma flatmap_map_at (m: Z[A]) (b: B) : flatmap_at (λ a, {f a}) m b = m.support.sum (λ a, if f a = b then m a else 0)`
- Description: Specializes `flatmap_map` to a concrete time or element index.

### `map_apply`

- Kind: `lemma`
- Source: `src/zset.lean:421`
- Signature: `lemma map_apply (m: Z[A]) (b: B) : zset.map f m b = m.support.sum (λ a, if f a = b then m a else 0)`
- Description: Gives the pointwise evaluation rule for `map`.

### `map_linear`

- Kind: `theorem`
- Source: `src/zset.lean:428`
- Signature: `theorem map_linear (f: A → B) (m1 m2: Z[A]) : zset.map f (m1 + m2) = zset.map f m1 + zset.map f m2`
- Description: Shows that `map` is linear.

### `map_is_card`

- Kind: `lemma`
- Source: `src/zset.lean:434`
- Signature: `lemma map_is_card (s: finset A) : ∀ b, zset.map f (zset.from_set s) b = (s.val.map f).count b`
- Description: Relates mapped-set multiplicities to cardinalities of preimages.

### `sum_nonneg`

- Kind: `lemma`
- Source: `src/zset.lean:453`
- Signature: `lemma sum_nonneg (s: finset A) (f: A → ℤ) : (∀ x, x ∈ s → 0 ≤ f x) → 0 ≤ s.sum f`
- Description: Provides a supporting summation lemma about `sum_nonneg`.

### `sum_pos`

- Kind: `lemma`
- Source: `src/zset.lean:464`
- Signature: `lemma sum_pos (s: finset A) (f: A → ℤ) : (∃ a, a ∈ s ∧ 0 < f a) → (∀ x, x ∈ s → 0 ≤ f x) → 0 < s.sum f`
- Description: Shows that `sum` is positive or positivity-preserving.

### `map_at_nonneg`

- Kind: `lemma`
- Source: `src/zset.lean:479`
- Signature: `lemma map_at_nonneg (f: A → B) (m: Z[A]) (b: B) : is_bag m → 0 ≤ flatmap_at (λ a, {f a}) m b`
- Description: Provides a supporting lemma about `map` in the stated setting.

### `map_at_pos`

- Kind: `lemma`
- Source: `src/zset.lean:491`
- Signature: `lemma map_at_pos (f: A → B) (m: Z[A]) (b: B) : is_bag m → (0 < flatmap_at (λ a, {f a}) m b ↔ ∃ a, a ∈ m ∧ f a = b)`
- Description: Shows that `map_at` is positive or positivity-preserving.
