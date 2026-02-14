#!/usr/bin/env python3
"""RLM Ralph Loop Multi-Agent Orchestration Engine.

A parsimonious implementation of recursive semantic decomposition:
- atomize context into weighted terms
- build similarity and dependency graphs
- recursively split low-cohesion clusters
- extract macro dependencies, cycles, and bridges
- control loop via confidence and structural stability
- emit audit artifacts with explicit schemas and epoch history
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


DEFAULT_MIN_SIMILARITY = 0.05
DEFAULT_MAX_DEPTH = 4
DEFAULT_TARGET_CONFIDENCE = 0.95
DEFAULT_MAX_ITERATIONS = 4
DEFAULT_MAX_BRIDGES = 64
DEFAULT_PER_DOC_ATOMS = 64
DEFAULT_MAX_FILE_CHARS = 250_000
DEFAULT_STABILITY_PATIENCE = 2
DEFAULT_MIN_BRIDGE_WEIGHT = 0.68
DEFAULT_DECOMPOSE_MIN_SIZE = 2
SCHEMA_VERSION = "rlm-index-v1"
MARKOV_ALPHA = 0.85
MARKOV_STABLE_EPSILON = 0.002

STOPWORDS = {
    "a", "an", "the", "and", "or", "if", "of", "to", "in", "for", "on",
    "by", "with", "without", "is", "are", "was", "were", "be", "been",
    "being", "it", "this", "that", "these", "those", "we", "you", "he", "she",
    "they", "them", "i", "me", "my", "our", "your", "yours", "their", "its",
    "not", "no", "can", "could", "should", "would", "will", "may", "must",
    "also", "from", "at", "as", "into", "about", "then", "than", "so", "but",
    "while", "after", "before", "inside", "outside", "there", "where", "which",
    "who", "whom", "what", "how", "why", "however", "such", "every", "each",
    "other", "only", "both", "using", "used", "use", "has", "have", "had", "do",
    "did", "does", "done", "theirs", "one", "two", "new", "newest", "previous",
    "next", "hereby", "again", "within", "thus",
}

EXCLUDE_DIR_PATTERNS = {
    ".git",
    ".venv",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
    "dist",
    "build",
    "__pycache__",
    ".next",
    ".turbo",
    "vendor",
}

EXTENSION_INCLUDE = {
    ".md", ".txt", ".py", ".ts", ".tsx", ".js", ".jsx", ".json", ".yaml", ".yml",
    ".toml", ".ini", ".cfg", ".sh", ".bash", ".zsh", ".java", ".cpp", ".c", ".h",
    ".go", ".rb", ".php", ".rs", ".sql", ".cs",
}

KNOWN_ROOT_FILENAMES = {"README", "LICENSE", "Makefile", "Dockerfile", "Pipfile"}


def _slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9._-]+", "-", value)
    value = re.sub(r"-{2,}", "-", value).strip("-")
    return value[:72] or "unit"


def _bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


@dataclass
class Thoughtbox:
    epoch: int
    observations: List[str]
    assumptions: List[str]
    decisions: List[str]
    risks: List[str]
    open_questions: List[str]


@dataclass
class RunConfig:
    input_dir: Path
    output_dir: Path
    min_similarity: float = DEFAULT_MIN_SIMILARITY
    max_depth: int = DEFAULT_MAX_DEPTH
    target_confidence: float = DEFAULT_TARGET_CONFIDENCE
    include_dot_git: bool = False
    mode: str = "orchestration"
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    max_bridges: int = DEFAULT_MAX_BRIDGES
    max_file_chars: int = DEFAULT_MAX_FILE_CHARS
    per_doc_atoms: int = DEFAULT_PER_DOC_ATOMS
    stability_patience: int = DEFAULT_STABILITY_PATIENCE
    min_bridge_weight: float = DEFAULT_MIN_BRIDGE_WEIGHT


@dataclass
class EpochState:
    epoch: int
    module_count: int
    module_signature: str
    confidence: float
    bridge_count: int
    cycle_count: int
    min_similarity: float
    status: str
    observations: List[str]
    risks: List[str]
    decisions: List[str]


def _safe_read_file(path: Path, max_chars: int) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[:max_chars]
    except OSError:
        return ""


def _normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = text.replace("\r\n", "\n")
    return re.sub(r"\n{3,}", "\n\n", text.strip())


def _tokenize(text: str) -> List[str]:
    raw = re.findall(r"[A-Za-z0-9_./-]+", text.lower())
    tokens: List[str] = []
    for token in raw:
        token = token.strip("._-")
        if not token or len(token) < 3:
            continue
        if token in STOPWORDS:
            continue
        if token.isdigit():
            continue
        tokens.append(token)
    return tokens


def _ngrams(tokens: List[str], n: int) -> Iterable[str]:
    if len(tokens) < n:
        return []
    for i in range(len(tokens) - n + 1):
        yield "_".join(tokens[i : i + n])


def _extract_atoms(text: str, per_doc_cap: int) -> Tuple[List[str], Dict[str, float]]:
    normalized = _normalize_text(text)
    tokens = _tokenize(normalized)
    singles = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    counts = Counter()
    counts.update(singles[:600])
    counts.update(_ngrams(singles, 2))
    counts.update(_ngrams(singles, 3))

    weighted: Dict[str, float] = {}
    for term, count in counts.items():
        term_len = len(term.split("_"))
        score = float(count) * (0.9 + 0.12 * term_len)
        if term_len == 1:
            score *= 0.6
        if any(ch.isdigit() for ch in term):
            score *= 0.8
        if score > 0.1:
            weighted[term] = round(score, 6)

    selected = [k for k, _ in sorted(weighted.items(), key=lambda kv: (-kv[1], kv[0]))][:per_doc_cap]
    return selected, weighted


def atom_of_thought_unit(text: str, per_doc_cap: int = DEFAULT_PER_DOC_ATOMS) -> Dict[str, object]:
    terms, scores = _extract_atoms(text, per_doc_cap=per_doc_cap)
    return {
        "atoms": terms,
        "scores": scores,
        "token_count": len(_tokenize(_normalize_text(text))),
    }


def thoughtbox_update(tb: Thoughtbox, note: str, bucket: str = "observations") -> None:
    if not note:
        return
    buckets = {
        "observations": tb.observations,
        "assumptions": tb.assumptions,
        "decisions": tb.decisions,
        "risks": tb.risks,
        "open_questions": tb.open_questions,
    }
    buckets[bucket].append(note)


def collect_source_units(root: Path, include_dot_git: bool = False, max_file_chars: int = DEFAULT_MAX_FILE_CHARS) -> Dict[str, str]:
    units: Dict[str, str] = {}
    if not root.exists():
        return units

    for candidate in root.rglob("*"):
        if not candidate.is_file():
            continue

        parts = set(candidate.parts)
        if any(ex in parts for ex in EXCLUDE_DIR_PATTERNS):
            continue
        if (not include_dot_git) and ".git" in parts:
            continue
        if candidate.suffix and candidate.suffix.lower() not in EXTENSION_INCLUDE:
            continue
        if not candidate.suffix and candidate.name not in KNOWN_ROOT_FILENAMES:
            continue

        data = _safe_read_file(candidate, max_chars=max_file_chars).strip()
        if not data:
            continue
        units[str(candidate.relative_to(root))] = data

    return units


def term_vector_map(units: Dict[str, str], per_doc_atoms: int) -> Dict[str, Dict[str, float]]:
    vectors: Dict[str, Dict[str, float]] = {}
    df: Counter[str] = Counter()
    tf_by_doc: Dict[str, Counter] = {}

    for path, text in units.items():
        atom_payload = atom_of_thought_unit(text, per_doc_cap=per_doc_atoms)
        term_tf = Counter(atom_payload["scores"])  # type: ignore[arg-type]
        tf_by_doc[path] = term_tf
        df.update(term_tf.keys())

    doc_count = max(1, len(units))
    for path, term_tf in tf_by_doc.items():
        vector: Dict[str, float] = {}
        for term, c in term_tf.items():
            idf = math.log((doc_count + 1) / (df[term] + 1)) + 1.0
            vector[term] = (1.0 + math.log(float(c))) * idf
        vectors[path] = vector
    return vectors


def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    dot = sum(v * b.get(k, 0.0) for k, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _semantic_edges(vectors: Dict[str, Dict[str, float]], threshold: float) -> Tuple[Dict[str, List[Tuple[str, float]]], List[Tuple[str, str, float, str]]]:
    paths = sorted(vectors.keys())
    adjacency: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    edge_rows: List[Tuple[str, str, float, str]] = []

    for i, source in enumerate(paths):
        for target in paths[i + 1 :]:
            score = _cosine(vectors[source], vectors[target])
            if score < threshold:
                continue
            adjacency[source].append((target, score))
            adjacency[target].append((source, score))
            edge_rows.append((source, target, score, "semantic"))
            edge_rows.append((target, source, score, "semantic"))

    return adjacency, edge_rows


def _candidate_file_index(units: Dict[str, str]) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    by_stem: Dict[str, List[str]] = defaultdict(list)
    by_path: Dict[str, str] = {}
    for rel in units:
        by_path[rel] = rel
        by_path[rel.replace("./", "")] = rel
        by_stem[Path(rel).stem].append(rel)
    return by_stem, by_path


def _extract_dependency_edges(units: Dict[str, str]) -> List[Tuple[str, str, float, str]]:
    by_stem, by_path = _candidate_file_index(units)
    patterns = [
        re.compile(r"^\s*import\s+[\w.,*\s]+\s+from\s+['\"]([^'\"]+)['\"]", re.MULTILINE),
        re.compile(r"^\s*import\s+['\"]([^'\"]+)['\"]", re.MULTILINE),
        re.compile(r"^\s*(?:const|let|var)\s+\w+\s*=\s*require\((['\"][^'\"]+['\"])", re.MULTILINE),
        re.compile(r"^\s*from\s+['\"]([^'\"]+)['\"]\s+import", re.MULTILINE),
        re.compile(r"^\s*include\s*\(\s*['\"]([^'\"]+)['\"]", re.MULTILINE),
        re.compile(r"^\s*#include\s*[<\"]([^>\"]+)[>\"]", re.MULTILINE),
    ]

    edge_rows: Set[Tuple[str, str, float, str]] = set()

    for source, text in units.items():
        for pat in patterns:
            for raw in pat.findall(text):
                dep = raw.split("?")[0].replace("\\", "/").strip().strip(".\"")
                if not dep:
                    continue
                dep = dep.rstrip("/")
                if dep.endswith((".py", ".ts", ".js", ".tsx", ".jsx")):
                    dep = dep[: dep.rfind(".")]

                if dep in by_path:
                    target = by_path[dep]
                    if target != source:
                        edge_rows.add((source, target, 0.92, "dependency"))
                    continue

                stem = Path(dep).stem
                if stem in by_stem:
                    for target in by_stem[stem]:
                        if target != source:
                            edge_rows.add((source, target, 0.78, "dependency"))
                            break
                    continue

                fallback = next((k for k in by_path if k.endswith(f"/{dep}")), None)
                if fallback and by_path[fallback] != source:
                    edge_rows.add((source, by_path[fallback], 0.5, "dependency"))

    return list(edge_rows)


def build_graph(units: Dict[str, str], min_similarity: float, per_doc_atoms: int) -> Tuple[Dict[str, List[Tuple[str, float]]], List[Tuple[str, str, float, str]], List[Dict[str, object]], Dict[str, Dict[str, float]]]:
    vectors = term_vector_map(units, per_doc_atoms=per_doc_atoms)
    semantic_adj, semantic_edges = _semantic_edges(vectors, min_similarity)
    dep_edges = _extract_dependency_edges(units)

    edge_rows = list({tuple(row) for row in (semantic_edges + dep_edges)})
    adjacency: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    for source, target, weight, _ in edge_rows:
        adjacency[source].append((target, weight))

    graph_rows = [
        {"source": source, "target": target, "weight": round(weight, 4), "type": edge_type}
        for source, target, weight, edge_type in edge_rows
    ]
    return adjacency, edge_rows, graph_rows, vectors


def _connected_components(nodes: List[str], adj: Dict[str, List[Tuple[str, float]]], threshold: float) -> List[List[str]]:
    seen: Set[str] = set()
    components: List[List[str]] = []
    for node in nodes:
        if node in seen:
            continue
        q = deque([node])
        seen.add(node)
        comp: List[str] = []
        while q:
            cur = q.popleft()
            comp.append(cur)
            for nxt, score in adj.get(cur, []):
                if score < threshold or nxt in seen:
                    continue
                seen.add(nxt)
                q.append(nxt)
        components.append(sorted(comp))
    return components


def _cohesion(nodes: List[str], adj: Dict[str, List[Tuple[str, float]]], threshold: float) -> float:
    if len(nodes) <= 1:
        return 1.0
    possible = len(nodes) * (len(nodes) - 1)
    edge_count = 0
    for node in nodes:
        for nxt, score in adj.get(node, []):
            if nxt in nodes and score >= threshold:
                edge_count += 1
    return edge_count / possible if possible else 0.0


def _recursive_decompose(nodes: List[str], adj: Dict[str, List[Tuple[str, float]]], weight: float, depth: int, max_depth: int) -> List[List[str]]:
    if len(nodes) <= DEFAULT_DECOMPOSE_MIN_SIZE or depth >= max_depth:
        return [sorted(nodes)]

    cohesion = _cohesion(nodes, adj, weight)
    if cohesion >= 0.24:
        return [sorted(nodes)]

    components = _connected_components(nodes, adj, min(weight * 1.15, 1.0))
    if len(components) == 1:
        return [sorted(nodes)]

    output: List[List[str]] = []
    for comp in components:
        output.extend(_recursive_decompose(comp, adj, min(weight * 1.05, 1.0), depth + 1, max_depth))
    return output


def _scc(nodes: List[str], edges: List[Tuple[str, str, float, str]]) -> List[List[str]]:
    adj: Dict[str, List[str]] = defaultdict(list)
    radj: Dict[str, List[str]] = defaultdict(list)
    for src, dst, _, _ in edges:
        adj[src].append(dst)
        radj[dst].append(src)

    visited: Set[str] = set()
    order: List[str] = []

    def dfs(u: str):
        visited.add(u)
        for nxt in adj.get(u, []):
            if nxt not in visited:
                dfs(nxt)
        order.append(u)

    for node in nodes:
        if node not in visited:
            dfs(node)

    comp_id: Dict[str, int] = {}
    comps: List[List[str]] = []

    def reverse_dfs(u: str, current: int, out: List[str]):
        comp_id[u] = current
        out.append(u)
        for nxt in radj.get(u, []):
            if nxt not in comp_id:
                reverse_dfs(nxt, current, out)

    for node in reversed(order):
        if node not in comp_id:
            out: List[str] = []
            reverse_dfs(node, len(comps), out)
            comps.append(sorted(out))
    return comps


def _topological_order(nodes: List[str], edges: List[Tuple[str, str, float, str]]) -> List[str]:
    indegree = {node: 0 for node in nodes}
    out: Dict[str, List[str]] = defaultdict(list)
    for src, dst, _, _ in edges:
        if src not in indegree or dst not in indegree:
            continue
        out[src].append(dst)
        indegree[dst] += 1

    q = deque([n for n, d in sorted(indegree.items()) if d == 0])
    order: List[str] = []
    while q:
        node = q.popleft()
        order.append(node)
        for nxt in out[node]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                q.append(nxt)

    # deterministic fallback for cycles
    for node in sorted(indegree):
        if node not in order:
            order.append(node)
    return order


def _markov_confidence(nodes: List[str], edges: List[Tuple[str, str, float, str]], max_steps: int = 200) -> Tuple[float, List[float]]:
    if not nodes:
        return 1.0, []

    out: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    for src, dst, weight, _ in edges:
        if src in nodes and dst in nodes:
            out[src].append((dst, weight))

    probs = {node: (idx + 1) / (len(nodes) * (len(nodes) + 1) / 2) for idx, node in enumerate(sorted(nodes))}
    alpha = MARKOV_ALPHA
    history: List[float] = []

    for step in range(max_steps):
        nxt: Dict[str, float] = {node: (1.0 - alpha) / len(nodes) for node in nodes}
        for src, current in probs.items():
            links = out.get(src, [])
            if not links:
                for node in nodes:
                    nxt[node] += alpha * current / len(nodes)
                continue
            total = sum(w for _, w in links) or 1.0
            for dst, weight in links:
                nxt[dst] += alpha * current * (weight / total)

        total = sum(nxt.values()) or 1.0
        for node in nxt:
            nxt[node] /= total

        delta = max(abs(nxt[node] - probs[node]) for node in nodes)
        history.append(delta)
        probs = nxt

        if step > 10 and all(h < MARKOV_STABLE_EPSILON for h in history[-8:]):
            break

    variance = sum(history) / len(history) if history else 1.0
    confidence = 1.0 - min(1.0, variance)
    return confidence, history


def _control_confidence(
    nodes: List[str],
    edges: List[Tuple[str, str, float, str]],
    cycles: List[Dict[str, object]],
) -> Tuple[float, float, List[float]]:
    if not nodes:
        return 1.0, 1.0, []
    if len(nodes) == 1:
        markov_confidence, history = 1.0, []
        return 1.0, markov_confidence, history

    markov_confidence, history = _markov_confidence(nodes, edges)
    active_nodes = set()
    for source, target, _, _ in edges:
        if source in nodes:
            active_nodes.add(source)
        if target in nodes:
            active_nodes.add(target)
    coverage = len(active_nodes) / len(nodes)
    cycle_penalty = 1.0 - min(1.0, len(cycles) / max(1, len(nodes)))
    confidence = (0.6 * markov_confidence) + (0.3 * coverage) + (0.1 * cycle_penalty)
    return min(1.0, max(0.0, confidence)), markov_confidence, history


def _is_better_state(
    candidate_confidence: float,
    candidate_module_count: int,
    candidate_cycle_count: int,
    candidate_markov_confidence: float,
    incumbent: Dict[str, object] | None,
) -> bool:
    if incumbent is None:
        return True

    incumbent_confidence = float(incumbent.get("confidence", -1.0))
    incumbent_modules = int(incumbent.get("module_count", 1 << 30))
    incumbent_cycles = int(incumbent.get("cycle_count", 1 << 30))
    incumbent_markov = float(incumbent.get("markov_confidence", -1.0))

    if candidate_confidence > incumbent_confidence:
        return True
    if not math.isclose(candidate_confidence, incumbent_confidence, abs_tol=1e-9):
        return False
    if candidate_module_count < incumbent_modules:
        return True
    if candidate_module_count > incumbent_modules:
        return False
    if candidate_cycle_count < incumbent_cycles:
        return True
    if candidate_cycle_count > incumbent_cycles:
        return False
    return candidate_markov_confidence > incumbent_markov


def _ordered_pair_key(left: str, right: str) -> str:
    return f"{left}|{right}" if left <= right else f"{right}|{left}"


def _build_module_ontology(vectors: Dict[str, Dict[str, float]], members: List[str], global_term_weights: Dict[str, float]) -> Dict[str, object]:
    merged = Counter()
    term_presence: Dict[str, int] = defaultdict(int)
    for member in members:
        local = vectors.get(member, {})
        merged.update(local)
        for term, weight in local.items():
            if weight > 0:
                term_presence[term] += 1

    core_terms = []
    for term, weight in sorted(merged.items(), key=lambda kv: (-kv[1], kv[0]))[:12]:
        core_terms.append({
            "term": term,
            "weight": round(float(weight), 4),
            "member_coverage": int(term_presence.get(term, 0)),
            "global_weight": round(float(global_term_weights.get(term, 0.0)), 4),
        })

    props = []
    local_term_counts = dict(merged)
    top_terms = [row["term"] for row in core_terms]
    for i, src_term in enumerate(top_terms):
        for j, dst_term in enumerate(top_terms):
            if i >= j:
                continue
            score = 0.0
            if src_term and dst_term:
                src_weight = local_term_counts.get(src_term, 0.0)
                dst_weight = local_term_counts.get(dst_term, 0.0)
                score = (min(src_weight, dst_weight) / max(src_weight + dst_weight, 1.0))
            if score > 0:
                support = 0
                for member in members:
                    local = vectors.get(member, {})
                    if local.get(src_term, 0.0) > 0 and local.get(dst_term, 0.0) > 0:
                        support += 1
                props.append({
                    "term_a": src_term,
                    "term_b": dst_term,
                    "coupling": round(score, 4),
                    "supporting_members": support,
                })

    member_coverage = [
        {
            "term": term,
            "weight": round(float(weight), 4),
            "member_coverage": int(coverage),
            "global_weight": round(float(global_term_weights.get(term, 0.0)), 4),
        }
        for term, weight, coverage in sorted(
            ((term, local_term_counts[term], term_presence.get(term, 0)) for term in local_term_counts),
            key=lambda row: (-row[1], -row[2], row[0]),
        )[:20]
    ]

    return {
        "core_terms": core_terms,
        "properties": sorted(props, key=lambda row: (-row["coupling"], row["term_a"], row["term_b"]))[:20],
        "member_coverage": member_coverage,
        "cardinality": len(merged),
    }


def _module_signature(modules: List[List[str]]) -> str:
    ordered = sorted(["|".join(m) for m in (sorted(x) for x in modules)])
    return "::".join(ordered)


def _make_run_dict(config: RunConfig) -> Dict[str, object]:
    return {
        "input_dir": str(config.input_dir),
        "output_dir": str(config.output_dir),
        "min_similarity": config.min_similarity,
        "max_depth": config.max_depth,
        "target_confidence": config.target_confidence,
        "include_dot_git": config.include_dot_git,
        "mode": config.mode,
        "max_iterations": config.max_iterations,
        "max_bridges": config.max_bridges,
        "max_file_chars": config.max_file_chars,
        "per_doc_atoms": config.per_doc_atoms,
        "stability_patience": config.stability_patience,
        "min_bridge_weight": config.min_bridge_weight,
    }


def _validate_manifest(manifest: Dict[str, object]) -> List[str]:
    errors: List[str] = []
    if not isinstance(manifest, dict):
        return ["manifest_not_object"]

    required = [
        "schema_version",
        "run",
        "modules",
        "bridges",
        "cycles",
        "graph",
        "topological_order",
        "epochs",
        "iterations_run",
        "confidence",
        "markov_confidence",
        "target_confidence",
        "selected_epoch",
        "selected_similarity",
        "thoughtbox",
        "signature",
        "status",
    ]
    for key in required:
        if key not in manifest:
            errors.append(f"missing:{key}")

    module_ids: Set[str] = set()
    if isinstance(manifest.get("modules"), list):
        for idx, module in enumerate(manifest["modules"]):  # type: ignore[index]
            if not isinstance(module, dict):
                errors.append(f"module_{idx}_not_object")
                continue
            required_module = ["module_id", "label", "members", "members_count", "cohesion", "atom_density", "outgoing_dependencies", "incoming_dependencies", "ontology", "bridge_refs"]
            for key in required_module:
                if key not in module:
                    errors.append(f"module_{idx}_missing:{key}")
            module_id = module.get("module_id")
            if isinstance(module_id, str):
                module_ids.add(module_id)
            if "members_count" in module and not isinstance(module.get("members_count"), int):
                errors.append(f"module_{idx}_invalid_members_count")
            if "cohesion" in module and not isinstance(module.get("cohesion"), (int, float)):
                errors.append(f"module_{idx}_invalid_cohesion")
            else:
                cohesion = float(module.get("cohesion", 0.0))
                if cohesion < 0.0 or cohesion > 1.0:
                    errors.append(f"module_{idx}_cohesion_out_of_range")
            if "atom_density" in module and not isinstance(module.get("atom_density"), (int, float)):
                errors.append(f"module_{idx}_invalid_atom_density")
            else:
                atom_density = float(module.get("atom_density", 0.0))
                if atom_density < 0.0:
                    errors.append(f"module_{idx}_atom_density_out_of_range")
            members = module.get("members")
            if not isinstance(members, list):
                errors.append(f"module_{idx}_invalid_members")
            elif "members_count" in module and module["members_count"] != len(members):
                errors.append(f"module_{idx}_members_count_mismatch")
            elif "members_count" in module and not isinstance(module["members_count"], int):
                errors.append(f"module_{idx}_members_count_not_int")
            if isinstance(module.get("outgoing_dependencies"), list):
                for eidx, edge in enumerate(module.get("outgoing_dependencies", [])):
                    if not isinstance(edge, dict):
                        errors.append(f"module_{idx}_outgoing_{eidx}_invalid")
            if isinstance(module.get("incoming_dependencies"), list):
                for eidx, edge in enumerate(module.get("incoming_dependencies", [])):
                    if not isinstance(edge, dict):
                        errors.append(f"module_{idx}_incoming_{eidx}_invalid")
            if "ontology" in module and not isinstance(module.get("ontology"), dict):
                errors.append(f"module_{idx}_invalid_ontology")
            if "bridge_refs" in module and not isinstance(module.get("bridge_refs"), list):
                errors.append(f"module_{idx}_invalid_bridge_refs")

    if isinstance(manifest.get("bridges"), list):
        bridges = manifest["bridges"]  # type: ignore[assignment]
        run = manifest.get("run", {})
        output_dir = Path(str(run.get("output_dir", ""))) if isinstance(run, dict) else None
        for idx, bridge in enumerate(bridges):  # type: ignore[index]
            if not isinstance(bridge, dict):
                errors.append(f"bridge_{idx}_not_object")
                continue
            required_bridge = ["bridge_id", "source_module", "target_module", "strength", "support", "notes", "justification", "bridge_file"]
            for key in required_bridge:
                if key not in bridge:
                    errors.append(f"bridge_{idx}_missing:{key}")
            if "source_module" in bridge and str(bridge["source_module"]) not in module_ids:
                errors.append(f"bridge_{idx}_source_unknown:{bridge['source_module']}")
            if "target_module" in bridge and str(bridge["target_module"]) not in module_ids:
                errors.append(f"bridge_{idx}_target_unknown:{bridge['target_module']}")
            if "support" in bridge and not isinstance(bridge.get("support"), int):
                errors.append(f"bridge_{idx}_invalid_support")
            if "strength" in bridge and not isinstance(bridge.get("strength"), (int, float)):
                errors.append(f"bridge_{idx}_invalid_strength")
            if "bridge_file" in bridge and output_dir is not None:
                bridge_file = Path(str(bridge["bridge_file"]))
                if not bridge_file.is_absolute():
                    bridge_file = output_dir / bridge_file
                if not bridge_file.exists():
                    errors.append(f"bridge_{idx}_missing_file:{bridge['bridge_file']}")

    run_graph = manifest.get("run")
    if not isinstance(run_graph, dict):
        errors.append("run_not_object")

    numeric_keys = {
        "confidence": (0.0, 1.0),
        "markov_confidence": (0.0, 1.0),
        "target_confidence": (0.0, 1.0),
    }
    for key, (low, high) in numeric_keys.items():
        if key in manifest and not isinstance(manifest[key], (int, float)):
            errors.append(f"{key}_invalid_type")
        if isinstance(manifest.get(key), (int, float)):
            value = float(manifest[key])
            if value < low or value > high:
                errors.append(f"{key}_out_of_range")

    for key in ("iterations_run", "selected_epoch"):
        if key in manifest and not isinstance(manifest.get(key), int):
            errors.append(f"{key}_invalid_type")

    if isinstance(manifest.get("selected_similarity"), (int, float)):
        similarity = float(manifest["selected_similarity"])
        if similarity < 0.0 or similarity > 1.0:
            errors.append("selected_similarity_out_of_range")

    if isinstance(manifest.get("topological_order"), list):
        seen: Set[str] = set()
        for idx, node in enumerate(manifest["topological_order"]):  # type: ignore[assignment]
            if not isinstance(node, str):
                errors.append(f"topology_{idx}_not_string")
                continue
            if node in seen:
                errors.append(f"topology_duplicate:{node}")
            seen.add(node)
            if node not in module_ids:
                errors.append(f"topology_unknown_node:{node}")

    if isinstance(manifest.get("cycles"), list):
        for idx, cycle in enumerate(manifest["cycles"]):  # type: ignore[assignment]
            if not isinstance(cycle, dict):
                errors.append(f"cycle_{idx}_not_object")
                continue
            cycle_nodes = cycle.get("nodes")
            if isinstance(cycle_nodes, list):
                for node in cycle_nodes:
                    if node not in module_ids:
                        errors.append(f"cycle_{idx}_node_unknown:{node}")

    if isinstance(manifest.get("graph"), list):
        for idx, edge in enumerate(manifest["graph"]):  # type: ignore[assignment]
            if not isinstance(edge, dict):
                errors.append(f"graph_{idx}_not_object")
                continue
            for key in ("source", "target", "type"):
                if key not in edge:
                    errors.append(f"graph_{idx}_missing:{key}")
            if "weight" in edge and not isinstance(edge.get("weight"), (int, float)):
                errors.append(f"graph_{idx}_invalid_weight")
            source = edge.get("source")
            target = edge.get("target")
            if source is not None and source not in module_ids:
                errors.append(f"graph_{idx}_unknown_source:{source}")
            if target is not None and target not in module_ids:
                errors.append(f"graph_{idx}_unknown_target:{target}")
            if isinstance(edge.get("weight"), (int, float)):
                weight = float(edge["weight"])
                if weight < 0.0 or weight > 1.0:
                    errors.append(f"graph_{idx}_weight_out_of_range")

    return errors


def run(config: RunConfig) -> Dict[str, object]:
    tb = Thoughtbox(
        epoch=0,
        observations=[],
        assumptions=[],
        decisions=[],
        risks=[],
        open_questions=[],
    )

    atlas: List[Dict[str, object]] = []

    source_units = collect_source_units(config.input_dir, include_dot_git=config.include_dot_git, max_file_chars=config.max_file_chars)
    if not source_units:
        thoughtbox_update(tb, "No analyzable files discovered.", "open_questions")
        no_source_epoch = asdict(EpochState(
            epoch=0,
            module_count=0,
            module_signature="",
            confidence=0.0,
            bridge_count=0,
            cycle_count=0,
            min_similarity=config.min_similarity,
            status="no-source",
            observations=list(tb.observations),
            risks=list(tb.risks),
            decisions=list(tb.decisions),
        ))
        payload = {
            "schema_version": SCHEMA_VERSION,
            "run": _make_run_dict(config),
            "status": "no-source",
            "modules": [],
            "bridges": [],
            "cycles": [],
            "graph": [],
            "thoughtbox": asdict(tb),
            "topological_order": [],
            "epochs": [no_source_epoch],
            "iterations_run": 1,
            "confidence": 0.0,
            "markov_confidence": 1.0,
            "target_confidence": config.target_confidence,
            "selected_epoch": 0,
            "selected_similarity": config.min_similarity,
            "selected_cycle_count": 0,
            "signature": "",
        }
        schema_errors = _validate_manifest(payload)
        if schema_errors:
            payload["schema_errors"] = schema_errors
        _safe_write(config.output_dir / "graph.json", json.dumps(payload, indent=2))
        _safe_write(config.output_dir / "state.json", json.dumps({"run": _make_run_dict(config), "atlas": [asdict(tb)]}, indent=2))
        _safe_write(config.output_dir / "atlas.jsonl", json.dumps(asdict(tb)) + "\n")
        _safe_write(config.output_dir / "index.md", "# RLM Orchestration Index\n\nNo source units found.\n")
        return payload

    vectors: Dict[str, Dict[str, float]] = term_vector_map(source_units, per_doc_atoms=config.per_doc_atoms)
    global_term_weights = Counter()
    for vector in vectors.values():
        global_term_weights.update(vector)
    dependency_edges = _extract_dependency_edges(source_units)
    dependency_signature = _module_signature([list(edge[:2]) for edge in dependency_edges])

    last_signature = None
    stability = 0
    best_state: Dict[str, object] | None = None
    current_similarity = max(0.01, config.min_similarity)
    bridge_support: Counter[str] = Counter()

    all_nodes = sorted(source_units.keys())

    for epoch in range(1, max(1, config.max_iterations) + 1):
        tb.epoch = epoch

        semantic_adjacency, semantic_edges = _semantic_edges(vectors, current_similarity)
        epoch_edge_rows = list({tuple(row) for row in (semantic_edges + dependency_edges)})
        adjacency: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        for source, target, weight, _ in sorted(epoch_edge_rows):
            adjacency[source].append((target, weight))
        for source in adjacency:
            adjacency[source].sort(key=lambda row: (row[0], -row[1]))
        graph_rows = [
            {"source": source, "target": target, "weight": round(weight, 4), "type": edge_type}
            for source, target, weight, edge_type in epoch_edge_rows
        ]

        modules = _recursive_decompose(all_nodes, adjacency, current_similarity, 0, config.max_depth)
        module_to_nodes: Dict[str, List[str]] = {}
        for idx, members in enumerate(modules):
            module_id = _slugify(f"module-{idx:03d}-{len(members)}")
            module_to_nodes[module_id] = sorted(members)
        node_to_module = {node: module_id for module_id, nodes in module_to_nodes.items() for node in nodes}

        macro_edges: List[Tuple[str, str, float, str]] = []
        for src, dst, weight, edge_type in epoch_edge_rows:
            src_module = node_to_module.get(src)
            dst_module = node_to_module.get(dst)
            if not src_module or not dst_module or src_module == dst_module:
                continue
            macro_edges.append((src_module, dst_module, weight, edge_type))

        macro_nodes = sorted(module_to_nodes.keys())
        macro_edges = sorted({(s, t, w, et) for s, t, w, et in macro_edges}, key=lambda row: (row[0], row[1], -row[2], row[3]))
        for src, dst, weight, _ in macro_edges:
            if weight >= config.min_bridge_weight:
                bridge_support[_ordered_pair_key(src, dst)] += 1

        cycles = [
            {"id": _slugify("cycle-" + "-".join(comp)), "nodes": comp}
            for comp in _scc(macro_nodes, macro_edges)
            if len(comp) > 1
        ]
        topo = _topological_order(macro_nodes, macro_edges)
        confidence, markov_confidence, history = _control_confidence(macro_nodes, macro_edges, cycles)
        signature = _module_signature(modules)
        module_count = len(module_to_nodes)

        if signature == last_signature:
            stability += 1
        else:
            stability = 0
            last_signature = signature

        thoughtbox_update(tb, f"epoch={epoch}, modules={module_count}, macro_nodes={len(macro_nodes)}, confidence={round(confidence, 5)}")

        atlas.append(asdict(EpochState(
            epoch=epoch,
            module_count=module_count,
            module_signature=signature,
            confidence=round(confidence, 5),
            bridge_count=0,
            cycle_count=len(cycles),
            min_similarity=round(current_similarity, 6),
            status="running",
            observations=list(tb.observations),
            risks=list(tb.risks),
            decisions=list(tb.decisions),
        )))

        if _is_better_state(
            candidate_confidence=confidence,
            candidate_module_count=module_count,
            candidate_cycle_count=len(cycles),
            candidate_markov_confidence=markov_confidence,
            incumbent=best_state,
        ):
            bridge_candidates = sorted(macro_edges, key=lambda row: (-row[2], row[0], row[1]))
            best_state = {
                "epoch": epoch,
                "module_to_nodes": module_to_nodes,
                "macro_edges": macro_edges,
                "topo": topo,
                "cycles": cycles,
                "confidence": confidence,
                "history": history,
                "markov_confidence": markov_confidence,
                "bridge_candidates": bridge_candidates,
                "signature": signature,
                "macro_nodes": macro_nodes,
                "cycle_count": len(cycles),
                "status": "best",
                "adjacency": adjacency,
                "graph_rows": graph_rows,
                "dependency_signature": dependency_signature,
                "min_similarity": current_similarity,
                "bridge_support": dict(bridge_support),
                "module_count": module_count,
            }

        if confidence >= config.target_confidence:
            thoughtbox_update(tb, f"target confidence reached at epoch {epoch}", "decisions")
            break

        if stability >= config.stability_patience:
            thoughtbox_update(tb, "structure stabilised; control loop exited", "decisions")
            break

        # control policy: reduce threshold when confidence low to permit more macro wiring
        residual = max(0.0, config.target_confidence - confidence)
        next_similarity = max(0.01, current_similarity * (1.0 - min(0.45, residual)))
        if math.isclose(next_similarity, current_similarity, rel_tol=1e-4):
            next_similarity = max(0.01, current_similarity - 0.01)
        current_similarity = next_similarity

    if best_state is None:
        raise RuntimeError("No candidate state selected. Check input corpus and dependency settings")

    module_to_nodes = best_state["module_to_nodes"]  # type: ignore[assignment]
    selected_similarity = best_state["min_similarity"]  # type: ignore[assignment]
    macro_edges = best_state["macro_edges"]  # type: ignore[assignment]
    cycles = best_state["cycles"]  # type: ignore[assignment]
    topo = best_state["topo"]  # type: ignore[assignment]
    selected_bridge_support = best_state["bridge_support"]  # type: ignore[assignment]
    selected_adjacency = best_state["adjacency"]  # type: ignore[assignment]
    selected_epoch = best_state["epoch"]  # type: ignore[assignment]

    bridge_candidates = best_state["bridge_candidates"]  # type: ignore[assignment]
    bridges: List[Dict[str, object]] = []

    selected_edges: List[Tuple[str, str, float, str, int]] = []
    used_pairs: Set[Tuple[str, str]] = set()
    for src, dst, weight, edge_type in bridge_candidates:
        if weight < config.min_bridge_weight or src == dst:
            continue
        pair = tuple(sorted((src, dst)))
        if pair in used_pairs:
            continue
        used_pairs.add(pair)
        support_count = int(selected_bridge_support.get(_ordered_pair_key(src, dst), 0))
        selected_edges.append((src, dst, weight, edge_type, support_count))
        if len(selected_edges) >= config.max_bridges:
            break

    for src, dst, weight, edge_type, support_count in selected_edges:
        if src == dst:
            continue
        bridge_id = f"{src}--{dst}"
        file_name = f"{_slugify(bridge_id)}.md"
        bridge_file = config.output_dir / "bridges" / file_name
        bridge_payload = {
            "bridge_id": bridge_id,
            "source_module": src,
            "target_module": dst,
            "strength": round(float(weight), 4),
            "support": support_count,
            "notes": [
                "Cross-module evidence bridge",
                "High-confidence semantic or dependency coupling preserved to avoid over-fragmentation",
            ],
            "justification": [f"edge_type:{edge_type}", f"selected_epoch:{selected_epoch}", f"persistence:{support_count}"],
            "bridge_file": str(bridge_file.relative_to(config.output_dir)),
        }
        _safe_write(bridge_file, json.dumps(bridge_payload, indent=2))

        if config.mode != "analysis":
            _safe_write(
                config.output_dir / "modules" / src / f"bridge-to-{_slugify(dst)}.md",
                f"../bridges/{file_name}\n",
            )
            _safe_write(
                config.output_dir / "modules" / dst / f"bridge-to-{_slugify(src)}.md",
                f"../bridges/{file_name}\n",
            )
        bridges.append(bridge_payload)

    module_defs: List[Dict[str, object]] = []
    for module_id, members in sorted(module_to_nodes.items(), key=lambda row: row[0]):
        members = sorted(members)
        outgoing = sorted(
            [{"to": dst, "weight": round(weight, 4), "type": edge_type}
             for src, dst, weight, edge_type in macro_edges if src == module_id],
            key=lambda row: (-row["weight"], row["to"], row["type"]),
        )
        incoming = sorted(
            [{"from": src, "weight": round(weight, 4), "type": edge_type}
             for src, dst, weight, edge_type in macro_edges if dst == module_id],
            key=lambda row: (-row["weight"], row["from"], row["type"]),
        )
        members_set = set(members)
        local_adjacency: Dict[str, List[Tuple[str, float]]] = {
            source: [(target, score) for target, score in selected_adjacency.get(source, []) if target in members_set]
            for source in members
        }
        cohesion = _cohesion(members, local_adjacency, selected_similarity)
        ontology = _build_module_ontology(vectors, members, {k: float(v) for k, v in global_term_weights.items()})
        module_payload = {
            "module_id": module_id,
            "label": f"module {module_id}",
            "members": members,
            "members_count": len(members),
            "cohesion": round(cohesion, 6),
            "atom_density": sum(len(vectors[member]) for member in members),
            "outgoing_dependencies": outgoing,
            "incoming_dependencies": incoming,
            "ontology": ontology,
            "bridge_refs": [b for b in bridges if b["source_module"] == module_id or b["target_module"] == module_id],
        }
        module_dir = config.output_dir / "modules" / module_id
        _safe_write(module_dir / "README.md", json.dumps(module_payload, indent=2))
        _safe_write(module_dir / "files.md", "\n".join(f"- `{m}`" for m in members) + "\n")
        _safe_write(module_dir / "atoms.json", json.dumps({m: sorted(vectors[m].keys()) for m in members}, indent=2))
        module_defs.append(module_payload)

    # update atlas with final bridge count
    for row in atlas:
        row["bridge_count"] = len(bridges)
        epoch_idx = row.get("epoch")
        if epoch_idx == selected_epoch:
            row["status"] = "selected"
        elif row.get("status") == "running":
            row["status"] = "completed"

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run": _make_run_dict(config),
        "modules": module_defs,
        "bridges": bridges,
        "cycles": cycles,
        "graph": [{"source": s, "target": t, "weight": round(w, 4), "type": et} for s, t, w, et in macro_edges],
        "topological_order": topo,
        "epochs": atlas,
        "iterations_run": len(atlas),
        "confidence": round(float(best_state["confidence"]), 5),
        "markov_confidence": round(float(best_state.get("markov_confidence", 0.0)), 5),
        "target_confidence": config.target_confidence,
        "selected_epoch": selected_epoch,
        "selected_similarity": selected_similarity,
        "selected_cycle_count": len(cycles),
        "status": "complete",
        "thoughtbox": asdict(tb),
        "signature": best_state["signature"],
    }

    schema_errors = _validate_manifest(manifest)
    if schema_errors:
        thoughtbox_update(tb, f"schema_validation_failures={len(schema_errors)}", "risks")
        manifest["schema_errors"] = schema_errors

    _safe_write(config.output_dir / "graph.json", json.dumps(manifest, indent=2))
    _safe_write(config.output_dir / "state.json", json.dumps({"run": _make_run_dict(config), "atlas": atlas, "thoughtbox": asdict(tb)}, indent=2))
    _safe_write(config.output_dir / "atlas.jsonl", "\n".join(json.dumps(row) for row in atlas) + "\n")

    index_lines = [
        "# RLM Orchestration Index",
        "",
        f"- status: {manifest['status']}",
        f"- confidence: {manifest['confidence']} (target {manifest['target_confidence']})",
        f"- markov confidence: {manifest['markov_confidence']}",
        f"- selected epoch: {selected_epoch}",
        f"- iterations run: {manifest['iterations_run']}",
        f"- modules: {len(module_defs)}",
        f"- bridges: {len(bridges)}",
        f"- cycles: {len(cycles)}",
        "",
        "## Topological order",
        "```",
        *topo,
        "```",
        "",
        "## Modules",
    ]
    for module in module_defs:
        index_lines.append(f"- `{module['module_id']}` ({module['members_count']} members)")

    index_lines.extend(["", "## Bridges", ""])
    for bridge in bridges:
        index_lines.append(f"- {bridge['source_module']} ⇄ {bridge['target_module']} — {bridge['bridge_file']} (w={bridge['strength']:.3f})")

    if cycles:
        index_lines.extend(["", "## Cycles", ""])
        for cycle in cycles:
            index_lines.append(f"- {cycle['id']} => {', '.join(cycle['nodes'])}")

    index_lines.extend(["", "## Control notes", ""])
    index_lines.extend([f"- {x}" for x in tb.observations[:24]])
    index_lines.extend(["", "## Risks", ""])
    if tb.risks:
        index_lines.extend([f"- {x}" for x in tb.risks])
    else:
        index_lines.append("- none")

    _safe_write(config.output_dir / "index.md", "\n".join(index_lines) + "\n")

    return manifest


def _safe_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RLM Ralph Loop Orchestration Engine")
    parser.add_argument("--input", default=".")
    parser.add_argument("--output", default=".rlm-out")
    parser.add_argument("--min-similarity", type=float, default=DEFAULT_MIN_SIMILARITY)
    parser.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH)
    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
    parser.add_argument("--target-confidence", type=float, default=DEFAULT_TARGET_CONFIDENCE)
    parser.add_argument("--include-dot-git", default="false")
    parser.add_argument("--mode", choices=["analysis", "orchestration"], default="orchestration")
    parser.add_argument("--max-bridges", type=int, default=DEFAULT_MAX_BRIDGES)
    parser.add_argument("--min-bridge-weight", type=float, default=DEFAULT_MIN_BRIDGE_WEIGHT)
    parser.add_argument("--stability-patience", type=int, default=DEFAULT_STABILITY_PATIENCE)
    parser.add_argument("--max-file-chars", type=int, default=DEFAULT_MAX_FILE_CHARS)
    parser.add_argument("--per-doc-atoms", type=int, default=DEFAULT_PER_DOC_ATOMS)
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    config = RunConfig(
        input_dir=Path(args.input).resolve(),
        output_dir=Path(args.output).resolve(),
        min_similarity=_clamp(args.min_similarity, 0.0, 1.0),
        max_depth=max(1, args.max_depth),
        target_confidence=_clamp(args.target_confidence, 0.0, 1.0),
        include_dot_git=_bool(args.include_dot_git),
        mode=args.mode,
        max_iterations=max(1, args.max_iterations),
        max_bridges=max(1, args.max_bridges),
        min_bridge_weight=_clamp(args.min_bridge_weight, 0.0, 1.0),
        stability_patience=max(1, args.stability_patience),
        max_file_chars=max(1, args.max_file_chars),
        per_doc_atoms=max(1, args.per_doc_atoms),
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = run(config)
    print(json.dumps({
        "status": manifest.get("status", "ok"),
        "modules": len(manifest.get("modules", [])),
        "bridges": len(manifest.get("bridges", [])),
        "cycles": len(manifest.get("cycles", [])),
        "confidence": manifest.get("confidence", 0.0),
        "target": manifest.get("target_confidence", config.target_confidence),
    }, indent=2))


if __name__ == "__main__":
    main()
