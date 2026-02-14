import json
import subprocess
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "rlm_orchestrator.py"


def _run_orchestrator(input_dir: Path, output_dir: Path, args: list[str] | None = None) -> dict:
    command = [
        "python3",
        str(SCRIPT),
        "--input",
        str(input_dir),
        "--output",
        str(output_dir),
        "--max-iterations",
        "3",
    ]
    if args:
        command.extend(args)

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise AssertionError(f"orchestrator failed: {result.stderr}\nstdout={result.stdout}")

    payload = json.loads((output_dir / "graph.json").read_text(encoding="utf-8"))
    return payload


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class TestRLMOrchestrator(unittest.TestCase):
    def test_orchestration_e2e_basic(self):
        with tempfile.TemporaryDirectory() as root_dir:
            root = Path(root_dir)
            _write(root / "a.md", "# A\\nCore context: graph topology and state transitions.\\n")
            _write(root / "b.md", "# B\\nDepends on A module semantics and graph edges.\\n")
            _write(root / "c.py", "import json\\nVALUE = 'context'\\n")
            out = Path(tempfile.mkdtemp())

            payload = _run_orchestrator(root, out, ["--mode", "orchestration", "--target-confidence", "0.55"])
            self.assertEqual(payload["status"], "complete")
            self.assertEqual(payload["schema_version"], "rlm-index-v1")
            self.assertTrue(payload["modules"])
            self.assertIn("graph", payload)
            self.assertTrue(payload["topological_order"])
            self.assertTrue((out / "index.md").exists())
            self.assertTrue(payload["selected_epoch"] >= 1)
            self.assertGreaterEqual(float(payload["confidence"]), 0.0)
            self.assertLessEqual(float(payload["confidence"]), 1.0)
            manifest = json.loads((out / "graph.json").read_text(encoding="utf-8"))
            self.assertIn("iterations_run", manifest)
            self.assertIn("markov_confidence", manifest)
            self.assertGreaterEqual(int(manifest["iterations_run"]), 1)
            self.assertNotIn("schema_errors", manifest)
            module_ids = {module["module_id"] for module in manifest["modules"]}
            self.assertTrue(set(manifest["topological_order"]).issubset(module_ids))
            selected_rows = [row for row in manifest["epochs"] if row["epoch"] == manifest["selected_epoch"]]
            self.assertEqual(len(selected_rows), 1)
            self.assertEqual(selected_rows[0]["status"], "selected")

            manifest = json.loads((out / "graph.json").read_text(encoding="utf-8"))
            self.assertIn("iterations_run", manifest)
            self.assertIn("markov_confidence", manifest)
            self.assertGreaterEqual(int(manifest["iterations_run"]), 1)
            for module in manifest["modules"]:
                self.assertTrue({"module_id", "members", "members_count", "ontology", "cohesion"}.issubset(module.keys()))
                ontology = module["ontology"]
                self.assertIn("member_coverage", ontology)
                self.assertIsInstance(ontology["member_coverage"], list)
                module_dir = out / "modules" / module["module_id"]
                self.assertTrue((module_dir / "README.md").exists())
                self.assertTrue((module_dir / "files.md").exists())
                self.assertTrue((module_dir / "atoms.json").exists())
            for bridge in manifest["bridges"]:
                self.assertIn("support", bridge)
                self.assertTrue((out / bridge["bridge_file"]).exists())
            bridge_pairs = []
            for bridge in manifest["bridges"]:
                pair = tuple(sorted((bridge["source_module"], bridge["target_module"])))
                bridge_pairs.append(pair)
            self.assertEqual(len(bridge_pairs), len(set(bridge_pairs)))
            self.assertIn("thoughtbox", manifest)
            for module in manifest["modules"]:
                ontology = module["ontology"]
                self.assertIn("core_terms", ontology)
                self.assertIn("properties", ontology)
                for prop in ontology["properties"]:
                    self.assertIn("supporting_members", prop)

    def test_empty_input_returns_no_source(self):
        with tempfile.TemporaryDirectory() as root_dir:
            in_dir = Path(root_dir) / "empty"
            in_dir.mkdir()
            out = Path(tempfile.mkdtemp())
            payload = _run_orchestrator(in_dir, out, ["--mode", "analysis"])
            self.assertEqual(payload["status"], "no-source")
            self.assertEqual(payload["modules"], [])
            self.assertIn("iterations_run", payload)
            self.assertIn("markov_confidence", payload)
            self.assertIn("selected_epoch", payload)
            self.assertIn("thoughtbox", payload)
            self.assertNotIn("schema_errors", payload)

    def test_analysis_mode_no_symlink_bridge_refs(self):
        with tempfile.TemporaryDirectory() as root_dir:
            root = Path(root_dir)
            _write(root / "a.py", "from .b import item\\n")
            _write(root / "b.py", "from .a import item\\n")
            out = Path(tempfile.mkdtemp())
            _run_orchestrator(root, out, ["--mode", "analysis", "--target-confidence", "0.3"])
            bridge_refs = list(out.glob("modules/**/bridge-to-*.md"))
            self.assertEqual(len(bridge_refs), 0)

    def test_cycle_detection_and_capsule(self):
        with tempfile.TemporaryDirectory() as root_dir:
            root = Path(root_dir)
            _write(root / "src.py", "state graph cycle mod context\\nimport mod\\n")
            _write(root / "mod.py", "from src import x\\nstate graph cycle mod context\\n")
            out = Path(tempfile.mkdtemp())
            payload = _run_orchestrator(root, out, ["--mode", "orchestration", "--target-confidence", "0.2", "--min-similarity", "0.01"])
            self.assertIn("cycles", payload)
            self.assertEqual(payload["status"], "complete")
            self.assertIsInstance(payload["graph"], list)

    def test_bridge_cap_and_validation(self):
        with tempfile.TemporaryDirectory() as root_dir:
            root = Path(root_dir)
            _write(root / "x.md", "alpha beta gamma graph\\n" * 20)
            _write(root / "y.md", "alpha beta graph dependency\\n" * 20)
            _write(root / "z.md", "alpha gamma dependency relation\\n" * 20)
            _write(root / "w.md", "unrelated unique tokens no overlap")
            out = Path(tempfile.mkdtemp())
            payload = _run_orchestrator(
                root,
                out,
                [
                    "--mode",
                    "orchestration",
                    "--target-confidence",
                    "0.25",
                    "--max-bridges",
                    "1",
                    "--min-bridge-weight",
                    "0.05",
                    "--min-similarity",
                    "0.01",
                ],
            )
            self.assertLessEqual(len(payload["bridges"]), 1)
            self.assertTrue((out / "state.json").exists())
            state = json.loads((out / "state.json").read_text(encoding="utf-8"))
            self.assertIn("thoughtbox", state)
            if payload["bridges"]:
                self.assertGreaterEqual(int(payload["bridges"][0]["support"]), 1)

    def test_clamped_control_parameters(self):
        with tempfile.TemporaryDirectory() as root_dir:
            root = Path(root_dir)
            _write(root / "a.md", "alpha beta gamma")
            out = Path(tempfile.mkdtemp())
            payload = _run_orchestrator(
                root,
                out,
                [
                    "--mode",
                    "analysis",
                    "--target-confidence",
                    "1.8",
                    "--min-similarity",
                    "-0.4",
                    "--min-bridge-weight",
                    "-0.2",
                    "--max-bridges",
                    "0",
                ],
            )
            self.assertGreaterEqual(payload["run"]["target_confidence"], 0.0)
            self.assertLessEqual(payload["run"]["target_confidence"], 1.0)
            self.assertGreaterEqual(payload["run"]["min_similarity"], 0.0)
            self.assertLessEqual(payload["run"]["min_similarity"], 1.0)
            self.assertGreaterEqual(payload["run"]["min_bridge_weight"], 0.0)
            self.assertLessEqual(payload["run"]["min_bridge_weight"], 1.0)
            self.assertGreaterEqual(payload["run"]["max_bridges"], 1)

    def test_deterministic_replay(self):
        with tempfile.TemporaryDirectory() as root_dir:
            root = Path(root_dir)
            _write(root / "a.md", "# A\\nGraph topology module semantics control path.\\n")
            _write(root / "b.md", "# B\\nDepends on A, graph semantics and module state.\\n")
            _write(root / "c.py", "from .a import value\\nVALUE = 'stable'\\n")
            out_1 = Path(tempfile.mkdtemp())
            out_2 = Path(tempfile.mkdtemp())

            first = _run_orchestrator(root, out_1, ["--mode", "orchestration", "--target-confidence", "0.55"])
            second = _run_orchestrator(root, out_2, ["--mode", "orchestration", "--target-confidence", "0.55"])

            self.assertEqual(first["signature"], second["signature"])
            self.assertEqual(first["topological_order"], second["topological_order"])
            self.assertEqual(first["selected_epoch"], second["selected_epoch"])
            self.assertEqual(len(first["epochs"]), len(second["epochs"]))

    def test_recursive_adaptation_drives_better_structure(self):
        with tempfile.TemporaryDirectory() as root_dir:
            root = Path(root_dir)
            _write(root / "a.md", "alpha beta gamma topology graph module architecture design")
            _write(root / "b.md", "alpha beta gamma topology module architecture system design")
            _write(root / "c.md", "alpha beta gamma topology engine system design control")
            _write(root / "d.md", "delta epsilon zeta isolated context")
            out = Path(tempfile.mkdtemp())
            payload = _run_orchestrator(
                root,
                out,
                [
                    "--mode",
                    "orchestration",
                    "--target-confidence",
                    "0.95",
                    "--min-similarity",
                    "0.5",
                    "--max-iterations",
                    "6",
                    "--stability-patience",
                    "6",
                ],
            )

            self.assertIn("epochs", payload)
            self.assertGreater(payload["selected_epoch"], 1)
            self.assertGreater(len(payload["epochs"]), 1)
            thresholds = [row["min_similarity"] for row in payload["epochs"]]
            self.assertGreater(len(set(thresholds)), 1)
            module_counts = [row["module_count"] for row in payload["epochs"]]
            self.assertNotEqual(module_counts[0], module_counts[-1])


if __name__ == "__main__":
    unittest.main()
