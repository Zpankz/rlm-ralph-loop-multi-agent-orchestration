# Refactor Skill

A production-ready Agent Skill that builds recursive dependency-aware modular architecture from raw context using an atom-of-thought pipeline, semantic graph analysis, adaptive decomposition, and cycle/bridge extraction.

This repository ships a standalone implementation for:

- extracting semantic atoms from files,
- building semantic + dependency graphs,
- recursively splitting low-cohesion modules,
- creating explicit bridge artifacts for weak/long-range dependencies,
- generating a final directed-acyclic module index,
- and emitting auditable graph/atlas artifacts with confidence tracking.

## Repository layout

- `SKILL.md`: skill contract (metadata + instructions) loaded by agent skill loaders.
- `scripts/`: orchestration engine and implementation.
- `tests/`: validation and regression tests.
- `references/`: schema and process references used by the skill.
- `.claude-plugin/marketplace.json`: Claude plugin marketplace descriptor.
- `.gitignore`: repo hygiene.

## Quickstart

```bash
python3 scripts/rlm_orchestrator.py --input <INPUT_DIR> --output <OUT_DIR>
```

## Install for Codex/Claude/Cursor/other agents

### Via ai-agent-skills (`npx skills`)

```bash
# Install this skill to all supported local agents
npx skills install Zpankz/refactor/refactor

# Install to only Claude's skills directory
npx skills install Zpankz/refactor/refactor --agent claude

# Dry-run preview
npx skills install Zpankz/refactor/refactor --agent claude --dry-run
```

## Claude plugin install

This repo includes a Claude marketplace manifest. After publishing, a Claude user can add the marketplace and install the plugin:

```bash
/plugin marketplace add https://github.com/Zpankz/refactor
/plugin install refactor@refactor-marketplace
```

## Supported runtime options

- `--mode analysis` for audit-only graph extraction.
- `--mode orchestration` for recursive refinement and module extraction.
- configurable thresholds: `--min-similarity`, `--target-confidence`, `--min-bridge-weight`, `--max-bridges`, and max depth settings.

## Security and scope

This project does not include any secrets, credentials, or deployment keys. It is deterministic and keeps model context output artifacts on disk for traceability.
