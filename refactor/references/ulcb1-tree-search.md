# ULCB1 tree search and convergence

## Concept

ULCB1 here is treated as a practical recursive discovery loop:

- U = Utility (informational gain),
- L = Linkage quality (edge coherence),
- C = Constraint conformance,
- B = Bridge necessity,
- 1 = one-pass deterministic ordering per epoch.

At each epoch, compute:
- mean utility gain from decomposition,
- mean linkage retention from parent to child modules,
- hard constraints satisfied or violated,
- bridge necessity.

Then recurse only when utility loss is positive and constraints remain enforceable.

## Stopping objective

The final state is accepted when:

- confidence is at or above target,
- no high-impact module split happened in the last two epochs,
- bridge churn is below threshold,
- cycle capsule count is stable.

When these fail, continue recursion if depth remains.

## Practical thresholds

- `target_confidence`: `0.95` default.
- `max_depth`: `4` default.
- `min_similarity`: `0.05` default, then raised during decomposition.

## Audit semantics

If `no-source` or repeated unstable decomposition is hit:
- keep current structure,
- preserve all decision traces,
- report risk for manual decision and rerun after constraint repair.

