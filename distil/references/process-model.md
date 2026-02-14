# Process model: Recursive RLM orchestration as a standalone system

## 1. Semantic primitives

### Atom-of-thought abstraction
The atomic unit is a concept tuple:
- source span (file)
- normalized term
- weight (local importance)
- provenance hash

No global meaning is inferred without edge context. The quality criterion is stable atoms per source.

### Thoughtbox state machine

Each epoch records:
- observations,
- assumptions,
- decisions,
- risks,
- open questions.

Persisting this state yields traceability and avoids non-stationary improvisation between recursion rounds.

### Infranodus-style graph layer

The graph layer consists of:
- semantic edges (similarity),
- dependency edges (references/imports),
- bridge edges (selected long-range high-importance interactions).

The graph is rebuilt each epoch with:
- updated semantic similarity thresholds,
- persistent dependency edges extracted from imports/references,
- current epoch graph statistics feeding control.

Control confidence is no longer a single fixed signal; it combines:
- convergence behavior on the macro graph,
- active-node coverage,
- cycle pressure after SCC extraction.

## 2. Recursive decomposition rule

Stop criteria for a module:

- module cohesion above threshold,
- module size low enough for direct stewardship,
- or recursion depth reached.

If none are true, split and continue.

## 3. Bridge policy

- only keep bridges above a stability threshold,
- include only edges with repeated cross-epoch activation or strong directional constraints,
- each bridge is explicit and traceable through a bridge document.

## 4. Directed acyclic formalization

1. detect SCCs,
2. collapse each SCC into a capsule node when needed,
3. topologically order macro nodes,
4. publish the order in `index.md`.

## 5. Confidence model

Confidence is Markov-chain convergence stability over macro transitions:

- high confidence = stable stationary probabilities and low iterative volatility,
- stable macro order and bounded bridge churn,
- low residual risk list.

Default target is `0.95`.

## 6. SOTA practices embedded

- Deterministic tie-breakers for reproducibility.
- bounded recursion to prevent over-fragmentation.
- explicit risk surfacing before destructive moves.
- no hidden external dependencies in the engine.
- clear provenance files for every generated artifact.
