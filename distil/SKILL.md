---
name: distil
description: "Execute recursive, dependency-aware context modularization and traversal for any code or document corpus. Use when you need to atomize context, classify and interconnect concepts, infer constraints and dependencies, create bridge files/symlinks/hyperedges, and produce a refined directed acyclic dependency model with progressive confidence gating."
---

# Distil

## Purpose

This skill provides a standalone best-practice pipeline for turning an undifferentiated context surface into a compact modular graph. It is designed for context-engineering workflows, repo rewrites, architecture consolidation, and knowledge-base triage where semantic structure and dependency correctness matter more than cosmetic refactoring.

The process is recursive. It keeps extracting structure, decomposing modules, creating bridges between distant but constrained dependencies, and re-running until a quality threshold is reached.

## Core operating model

Use this skill in two modes:

1. `analysis` to map and score the current state of context.
2. `orchestration` to map, refactor, and emit the final index.

In both modes, the same primitives apply:

- Atom-of-thought abstraction: break text into stable atomic semantic units.
- Thoughtbox state: persist cycle-local decisions, open questions, constraints, and unresolved risk.
- Infranodus-style graph analytics: build weighted term/coherence graphs, detect communities, and route cross-connections through bridge objects.
- Hyperedge bridging: preserve weak and long-range dependency semantics through explicit bridge files and references.
- Recursive DAG convergence: ensure the final structure is acyclic or minimized into explicit cycle capsules.

## When to use this skill

- When context is diffuse, highly duplicated, or weakly modularized.
- When dependencies are implicit and need explicit graph formalization.
- When you need a canonical modular map with bridges and recursion confidence.
- When you need a reusable orchestration framework that is minimally dependent and can run standalone.

## Trigger terms

Use when the request contains any of:
- recursive refactor
- context graph
- module decomposition
- dependency DAG
- bridge file
- concept extraction
- acyclic index
- "95%" or high-confidence optimization pass

## Entry requirements

1. The input corpus path (files/directories).
2. Optional output directory.
3. Optional constraints and hard exclusions.
4. Confidence target (default `0.95`).

## Standard process

1. Collect context units.
2. Extract atoms, tags, and concept signatures.
3. Build an undirected semantic graph and directional dependency graph.
4. Detect communities and recursively decompose unstable modules.
5. Detect cycles and create cycle capsules when necessary.
6. Create bridges for distant yet meaningful edges.
7. Emit module-level files, symlinks/references, and one main index.
8. Run a final Markov-style confidence pass and stop only when criteria are met or max recursion depth is reached.

## Directory layout produced

The orchestrated output defaults to:

- `index.md` main entry point and formal traversal order.
- `state.json` run state, thresholds, and convergence summary.
- `modules/<module-id>/README.md` module boundaries and local interfaces.
- `modules/<module-id>/atoms.json` canonical atoms for that module.
- `modules/<module-id>/files.md` linked source members.
- `bridges/<a>__<b>.md` cross-module bridge files.
- `bridges/symlinks/<module-id>/` optional symlink views.
- `graph.json` full graph dump and computed metrics.
- `atlas.jsonl` serialized thoughtbox states per recursion epoch.

If a repo style requires a different layout, mirror these artifacts with equivalent semantic roles.

## Standalone script integration

The bundled script is intended to be copied/adapted into any workspace:

- `/Users/mikhail/Projects/Context-Engineering/Distil/distil/scripts/rlm_orchestrator.py`

Use this as the default execution engine. It has zero mandatory third-party dependencies.

## Execution procedure

### 1) Atomization

For every file/segment:

1. Normalize Unicode spacing, comments, and line breaks.
2. Tokenize and extract n-grams in controlled windows.
3. Strip stop terms, noisy tokens, and identifiers that do not carry semantic weight.
4. Keep top weighted atoms by TF-IDF-like scoring.
5. Emit atom vectors and tags.

### 2) Thoughtbox pass

Maintain a persistent per-pass state object:

- `observations`: discovered constraints, naming conventions, domain-specific terms.
- `assumptions`: inferred links without high confidence.
- `decisions`: hard commitments and rationale.
- `risks`: cycle candidates, overloaded modules, low-confidence merges.
- `open_questions`: unresolved semantic ambiguity requiring explicit arbitration.

The thoughtbox object is always written to disk each recursion iteration so the process is auditable.

### 3) Graph build

Build two edge sets:

- semantic edges from similarity of atom vectors.
- dependency edges from explicit references/imports/mentions in text.

Store edge strength as floating scores in `[0,1]`.

### 4) Recursive decomposition

Use these rules recursively for each module:

1. compute modularity signal from edge density and interface breadth.
2. split on weak edge boundaries when cohesion is below threshold.
3. rerun atomization inside sub-modules as needed for local precision.
4. stop splitting when either:
   - cohesion is stable by threshold,
   - module size is already minimal,
   - or max recursion depth is hit.

### 5) DAG synthesis

1. Detect cycles in directed dependency graph.
2. Collapse each SCC into a cycle capsule when needed.
3. Sort capsules/modules topologically.
4. Re-link edges through capsules and bridges so traversal is formally acyclic at the macro level.

### 6) Bridge and hyperedge construction

For distanced but important edges:

1. emit bridge markdown documents with explicit rationale, source path, and target path.
2. emit references into both modules.
3. emit symlinks when filesystem permits, else canonical reference files.

Do not convert every weak edge to bridge. Reserve bridges for semantically dense relations that are:
- frequently co-activated,
- constrained by dependency,
- or repeated across traversal epochs.

### 7) Confidence and finalization

Run a Markov-chain traversal where probability mass moves along dependency direction.
Converge on a confidence measure across nodes and modules.

Stop when:
- `global_confidence >= configured target` (default `0.95`), or
- `max_iterations` reached,
- and no high-impact structural changes happened in the last two epochs.

### 8) Index formalization

Emit `index.md` with:
- module order (topological),
- bridge registry,
- rationale per module,
- unresolved risk log,
- action list for the next maintenance cycle.

## Minimal dependency policy

- Never hard-require external services.
- Keep all algorithms local and deterministic.
- Prefer deterministic tie-breakers.
- Preserve idempotence where possible.

## Script invocation

Default:

```bash
python3 scripts/rlm_orchestrator.py \
  --input /path/to/context \
  --output .rlm-out \
  --max-depth 4 \
  --min-similarity 0.05 \
  --target-confidence 0.95 \
  --include-dot-git false
```

For audit-only pass:

```bash
python3 scripts/rlm_orchestrator.py --input . --output .rlm-out --mode analysis
```

Important:

- Use `analysis` before destructive file moves.
- Keep source files in read-only mode during analysis.
- Materialize symlink-based bridges only during orchestration pass.

## Quality gates

Abort and revise if any of these are true:

- Core modules exceed 200 files without strong internal cohesion.
- More than 20% of edges become unidirectional contradictions.
- Multiple bridges point to stale paths.
- Cycles remain unresolved in macro graph.
- Confidence is under target by more than 5% with no structural trend.

## Failure handling

1. If semantic extraction is noisy, tighten token filters and rerun with domain stopwords.
2. If cycles proliferate, add explicit interface contracts first, then rerun.
3. If bridge count explodes, raise bridge threshold and keep only high-persistence edges.
4. If symlink creation fails, emit reference stubs with same semantics.

## Safe fallback behavior

- If recursive decomposition cannot improve structure, stop and produce a strict audit report instead of over-splitting.
- If path mapping is ambiguous, preserve the original source path and expose an explicit `manual_decision_required` flag.

## Reference map

- `/Users/mikhail/Projects/Context-Engineering/Distil/distil/scripts/rlm_orchestrator.py` is the implementation core.
- `/Users/mikhail/Projects/Context-Engineering/Distil/distil/references/process-model.md` explains the conceptual mapping.
- `/Users/mikhail/Projects/Context-Engineering/Distil/distil/references/ulcb1-tree-search.md` explains recursive stopping criteria.

## Non-goals

- It does not claim language/semantic perfection.
- It does not auto-run formatters, compilers, or tests.
- It does not guarantee no context loss; every split creates explicit rationale and provenance.

## Final requirement

The skill run is complete when the agent can answer:
- What are the modules?
- Why are they grouped?
- How do they depend?
- Where are the bridges and why?
- What part still requires a human decision?

Only then should the pipeline emit its final index and confidence summary.

## Schema-aware control model (v2)

- `graph.json` carries a strict contract: `schema_version`, `run`, `modules`, `bridges`, `cycles`, `graph`, `topological_order`, `epochs`, `confidence`, `target_confidence`, `selected_epoch`, `status`, `thoughtbox`.
- `run` is a compact machine control surface: thresholds, recursion limits, and bridge policy.
- `epochs` is the closed-loop audit trail for recursive refinement (with epoch state and transition status).
- Each module includes:
  - `module_id`, `label`, `members`, `members_count`
  - `cohesion`, `atom_density`
  - `outgoing_dependencies`, `incoming_dependencies`
  - `ontology` with `core_terms`, `properties`, `cardinality`
  - `bridge_refs`
- Each bridge includes:
  - `bridge_id`, `source_module`, `target_module`, `strength`
  - `notes`, `justification`, `bridge_file`

## E2E red-team test suite expectations

- Run `analysis` mode first, then `orchestration`.
- Verify outputs exist: `graph.json`, `index.md`, `state.json`, `atlas.jsonl`, `modules/*`, `bridges/*`.
- Verify recursive control adaptation:
  - graph rebuild occurs across epochs (`epochs[].min_similarity` should show adjustment when target is not met),
  - at least one epoch improves module signature or module counts before convergence,
  - final `selected_epoch` is selected by confidence progression and structural stability.
- Verify ontology and relational integrity:
  - each module ontology includes `core_terms[].member_coverage` and `properties[].supporting_members`,
  - each bridge includes `support` and `bridge_file` coherence.
- Verify failure-safety invariants:
  - no stale bridge references in `analysis` mode,
  - no path explosion beyond `max_bridges`,
  - deterministic output for repeated runs with same inputs.
- Verify control stability:
  - `selected_epoch >= 1`
  - `confidence` should monotonically improve or stabilize across final state
  - no stale duplicate bridges
- Verify schema checks:
  - `modules` entries satisfy required keys and types
  - `bridges` entries include pathable `bridge_file`
  - `topological_order` only contains known module identifiers
- Verify adversarial resilience:
  - empty input path => `no-source` status
  - cyclic dependency inputs produce cycle capsule entries
  - unsupported files are safely ignored
  - high-threshold run still emits analyzable structure without crashing

## Reference map

- `/Users/mikhail/Projects/Context-Engineering/Distil/distil/references/ontology-schema.md` for schema details.
- `/Users/mikhail/Projects/Context-Engineering/Distil/distil/references/process-model.md` for the conceptual model.
- `/Users/mikhail/Projects/Context-Engineering/Distil/distil/references/ulcb1-tree-search.md` for recursive control and stopping conditions.
