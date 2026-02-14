# Ontology and artifact schema

## Module schema

Each module entry in `graph.json`:

| field | type | description |
|---|---|---|
| `module_id` | string | Stable module identifier |
| `label` | string | Human label |
| `members` | array[string] | Source members assigned to the module |
| `members_count` | number | Cardinality of `members` |
| `cohesion` | number | Internal edge density proxy |
| `atom_density` | number | Count of term atoms across all members |
| `outgoing_dependencies` | array[object] | Edges from module to others |
| `incoming_dependencies` | array[object] | Edges from others to module |
| `ontology` | object | Conceptual schema |
| `bridge_refs` | array[object] | Bridge metadata touching the module |

### `ontology`

`ontology.core_terms`:
- `term`: string
- `weight`: number
- `global_weight`: number

`ontology.properties`:
- `term_a`: string
- `term_b`: string
- `coupling`: number
- `supporting_members`: number

`ontology.cardinality`: number

`ontology.member_coverage`:
- `term`: string
- `weight`: number
- `member_coverage`: number
- `global_weight`: number

## Bridge schema

Each bridge entry in `graph.json`:

- `bridge_id`: `source--target`
- `source_module`: module id
- `target_module`: module id
- `strength`: numeric edge score
- `support`: edge persistence count across control epochs
- `notes`: array[string]
- `justification`: array[string]
- `bridge_file`: path string to serialized bridge record

## Run-state schema

`run` includes:

- input/output paths
- control knobs (min-similarity, depth, iterations, confidence targets, thresholds)
- file extraction limits and policy flags

`epochs` rows include:
- `epoch`, `module_count`, `module_signature`, `confidence`, `bridge_count`, `cycle_count`, `status`

## Top-level manifest invariants

- `schema_version` must equal `rlm-index-v1`
- `status` expected values: `no-source` or `complete`
- `topological_order` contains only known modules
- `confidence` is numeric in [0,1]
