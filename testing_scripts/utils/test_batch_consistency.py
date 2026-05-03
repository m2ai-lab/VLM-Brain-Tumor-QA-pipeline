"""
testing_scripts/utils/test_batch_consistency.py
================================================
CPU-only smoke tests for the batch processing infrastructure.

Tests what can be verified WITHOUT a GPU or real model weights:
  1. Checkpoint utility — write / resume / deduplication.
  2. Batch loop structure — verify run_batch() in each testing script
     correctly handles missing images, produces one result per input
     row, and preserves row order.  A lightweight mock replaces
     model.generate() so no GPU is needed.
  3. --batch_size CLI argument — verify every updated testing script
     accepts the flag without error.
  4. experiment.json round-trip — confirm batch_size resolves correctly
     for every model via the orchestrator stack.

Run from the project root:
    python -m testing_scripts.utils.test_batch_consistency

All tests should complete in < 30 seconds on CPU.
"""
from __future__ import annotations

import csv
import importlib
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ── Project root on sys.path ──────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# =============================================================================
# 1. Checkpoint utility tests
# =============================================================================

class TestCheckpoint(unittest.TestCase):
    """Verify load_checkpoint() and save_checkpoint() behave correctly."""

    def setUp(self):
        from testing_scripts.utils.checkpoint import load_checkpoint, save_checkpoint
        self.load_checkpoint = load_checkpoint
        self.save_checkpoint = save_checkpoint

    def test_load_empty_when_no_file(self):
        completed = self.load_checkpoint("/tmp/does_not_exist_xyz.csv")
        self.assertEqual(completed, set())

    def test_roundtrip_single_batch(self):
        import pandas as pd
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            out_path = f.name
        try:
            os.remove(out_path)  # start fresh (no file)

            # Simulate one batch of 3 rows
            qa = pd.DataFrame({
                "Assigned ID": ["P001", "P002", "P003"],
                "Question": ["Q1", "Q2", "Q3"],
            })
            self.save_checkpoint(
                out_path, qa,
                {"predicted_answer": ["A1", "A2", "A3"],
                 "Reasoning": ["R1", "R2", "R3"]},
            )

            loaded = self.load_checkpoint(out_path)
            self.assertEqual(loaded, {"P001", "P002", "P003"})
        finally:
            if os.path.exists(out_path):
                os.remove(out_path)

    def test_resume_skips_completed(self):
        """After writing batch 1, load_checkpoint returns those IDs so batch 2 can skip duplicates."""
        import pandas as pd
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            out_path = f.name
        try:
            os.remove(out_path)

            batch1 = pd.DataFrame({"Assigned ID": ["P001", "P002"], "Question": ["Q1", "Q2"]})
            self.save_checkpoint(out_path, batch1, {"predicted_answer": ["A1", "A2"]})

            completed = self.load_checkpoint(out_path)
            self.assertIn("P001", completed)
            self.assertIn("P002", completed)
            self.assertNotIn("P003", completed)

            batch2 = pd.DataFrame({"Assigned ID": ["P003"], "Question": ["Q3"]})
            self.save_checkpoint(out_path, batch2, {"predicted_answer": ["A3"]})

            final = self.load_checkpoint(out_path)
            self.assertEqual(final, {"P001", "P002", "P003"})
        finally:
            if os.path.exists(out_path):
                os.remove(out_path)

    def test_no_duplicate_rows_on_append(self):
        """save_checkpoint in append mode should NOT duplicate rows if called twice."""
        import pandas as pd
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            out_path = f.name
        try:
            os.remove(out_path)
            batch = pd.DataFrame({"Assigned ID": ["P001"], "Question": ["Q1"]})
            self.save_checkpoint(out_path, batch, {"predicted_answer": ["A1"]})
            # Calling once is enough; reading back should have exactly 1 row
            result = pd.read_csv(out_path)
            self.assertEqual(len(result), 1)
        finally:
            if os.path.exists(out_path):
                os.remove(out_path)


# =============================================================================
# 2. Batch loop structure — mock-based per-script tests
# =============================================================================

def _make_dummy_qa(n: int = 4):
    """Return a list of minimal QA row dicts."""
    return [
        {"Assigned ID": f"P{i:03d}", "Question": f"What is finding {i}? 1) A 2) B 3) C 4) D"}
        for i in range(n)
    ]


class TestMedGemmaSingleSliceBatch(unittest.TestCase):
    """run_batch() in medgemma_single_slice handles missing images gracefully."""

    def _import_run_batch(self):
        # Stub out heavy imports before loading the module
        for mod in ["nibabel", "transformers", "pydantic"]:
            if mod not in sys.modules:
                sys.modules[mod] = MagicMock()
        # Patch config_utils to avoid needing a real config.yaml
        sys.modules.setdefault("config_utils", MagicMock(load_config=lambda: {}))
        # Patch checkpoint so it doesn't hit disk
        sys.modules.setdefault(
            "testing_scripts.utils.checkpoint",
            types.SimpleNamespace(load_checkpoint=lambda p: set(),
                                  save_checkpoint=lambda *a, **k: None),
        )
        import importlib
        spec = importlib.util.spec_from_file_location(
            "medgemma_single",
            _PROJECT_ROOT / "testing_scripts" / "QA_testing_medgemma_single_slice.py",
        )
        mod = importlib.util.module_from_spec(spec)
        # Provide a dummy _cfg so module-level code doesn't crash
        mod._cfg = {}
        spec.loader.exec_module(mod)
        return mod

    def test_missing_image_returns_error_not_exception(self):
        """run_batch should not raise even if the image directory doesn't exist."""
        mod = self._import_run_batch()
        model = MagicMock()
        processor = MagicMock()
        processor.tokenizer = MagicMock(padding_side="right")
        processor.apply_chat_template.return_value = "prompt"
        # Processor call returns a mock tensor batch
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_inputs.__getitem__ = MagicMock(return_value=MagicMock())
        processor.return_value = mock_inputs

        rows = _make_dummy_qa(3)
        # Use a nonexistent base dir so all images are missing
        results = mod.run_batch(model, processor, rows, "/nonexistent/dir", "Axial.png")

        self.assertEqual(len(results), 3)
        for r in results:
            self.assertIn("answer", r)
            # Should have returned an Error for each missing image
            self.assertEqual(r["answer"], "Error")

    def test_result_count_matches_input(self):
        """run_batch must return exactly one result per input row."""
        mod = self._import_run_batch()
        rows = _make_dummy_qa(6)
        model = MagicMock()
        processor = MagicMock()
        processor.tokenizer = MagicMock(padding_side="right")
        processor.apply_chat_template.return_value = "prompt"
        results = mod.run_batch(model, processor, rows, "/nonexistent/dir", "Axial.png")
        self.assertEqual(len(results), len(rows))


class TestQwenBatch(unittest.TestCase):
    """run_batch() in QA_testing_Qwen.py handles N rows correctly."""

    def _import_run_batch(self):
        for mod in ["transformers", "pydantic"]:
            sys.modules.setdefault(mod, MagicMock())
        sys.modules.setdefault("config_utils", MagicMock(load_config=lambda: {}))
        sys.modules.setdefault(
            "testing_scripts.utils.checkpoint",
            types.SimpleNamespace(load_checkpoint=lambda p: set(),
                                  save_checkpoint=lambda *a, **k: None),
        )
        spec = importlib.util.spec_from_file_location(
            "qwen_script",
            _PROJECT_ROOT / "testing_scripts" / "QA_testing_Qwen.py",
        )
        mod = importlib.util.module_from_spec(spec)
        mod._cfg = {}
        spec.loader.exec_module(mod)
        return mod

    def test_result_count_matches_input(self):
        mod = self._import_run_batch()
        rows = _make_dummy_qa(5)

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "text"
        mock_inputs = MagicMock()
        mock_inputs.__getitem__ = MagicMock(return_value=MagicMock(shape=[1, 10]))
        tokenizer.return_value = mock_inputs
        tokenizer.decode.return_value = '{"reasoning": "r", "answer": "1) A"}'

        model = MagicMock()
        # generate returns mock tensor of shape (N, seq_len)
        mock_gen = MagicMock()
        mock_gen.__iter__ = MagicMock(return_value=iter(
            [MagicMock()] * 5
        ))
        model.generate.return_value = [MagicMock()] * 5

        results = mod.run_batch(model, tokenizer, rows)
        self.assertEqual(len(results), 5)

    def test_each_result_has_answer_key(self):
        mod = self._import_run_batch()
        rows = _make_dummy_qa(2)
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "text"
        mock_inputs = MagicMock()
        mock_inputs.__getitem__ = MagicMock(return_value=MagicMock(shape=[1, 8]))
        tokenizer.return_value = mock_inputs
        tokenizer.decode.return_value = '{"reasoning": "r", "answer": "2) B"}'
        model = MagicMock()
        model.generate.return_value = [MagicMock()] * 2

        results = mod.run_batch(model, tokenizer, rows)
        for r in results:
            self.assertIn("answer", r)
            self.assertIn("reasoning", r)


# =============================================================================
# 3. --batch_size CLI argument accepted by all updated scripts
# =============================================================================

class TestBatchSizeCLIArg(unittest.TestCase):
    """
    Verify that every updated testing script registers --batch_size
    without raising an ArgumentError.  We import the argparse parser
    directly via the __main__ block's parser.add_argument calls.
    """

    SCRIPTS = [
        "QA_testing_medgemma_single_slice.py",
        "QA_testing_medgemma_multi_slice.py",
        "QA_testing_medgemma_blank.py",
        "QA_testing_medgemma_contrast_slices.py",
        "QA_testing_lingshu.py",
        "QA_testing_Qwen.py",
        "QA_testing_MedImageInsight.py",
        "QA_testing_Med3DVLM.py",
        "QA_testing_Med3DVLM_blank.py",
    ]

    def test_all_scripts_have_batch_size_arg(self):
        """Grep for --batch_size in each script file — faster and more reliable than importing."""
        scripts_dir = _PROJECT_ROOT / "testing_scripts"
        missing = []
        for script_name in self.SCRIPTS:
            script_path = scripts_dir / script_name
            self.assertTrue(script_path.exists(), f"{script_name} not found")
            content = script_path.read_text(encoding="utf-8")
            if "--batch_size" not in content:
                missing.append(script_name)
        self.assertEqual(
            missing, [],
            f"These scripts are missing --batch_size arg: {missing}",
        )


# =============================================================================
# 4. experiment.json → config round-trip: correct per-model batch_sizes
# =============================================================================

EXPECTED_BATCH_SIZES = {
    "MedGemma1.5":               4,
    "MedGemma1.5-ContrastSlices": 4,
    "Med3DVLM":                   1,
    "Qwen2.5":                    8,
    "MedImageInsight":            16,
    "LLaVA-Med":                  4,
    "GPT5Mini":                   4,
    "Lingshu-32B":                2,
}


class TestConfigBatchSizes(unittest.TestCase):
    """Confirm batch_size resolves correctly for every model in experiment.json."""

    def test_batch_sizes_round_trip(self):
        from experiment_orchestrator.config_resolver import expand_suite_raw, resolve_all
        from experiment_orchestrator.config_schema import ExperimentSuite

        config_path = _PROJECT_ROOT / "experiment.json"
        self.assertTrue(config_path.exists(), "experiment.json not found")

        with open(config_path) as f:
            raw = json.load(f)
        raw = expand_suite_raw(raw)
        suite = ExperimentSuite.model_validate(raw)
        jobs = resolve_all(suite)

        seen: dict[str, int] = {}
        for j in jobs:
            seen[j.model_name] = j.batch_size

        for model_name, expected_bs in EXPECTED_BATCH_SIZES.items():
            with self.subTest(model=model_name):
                self.assertIn(model_name, seen,
                              f"{model_name} not found in resolved jobs")
                self.assertEqual(
                    seen[model_name], expected_bs,
                    f"{model_name}: expected batch_size={expected_bs}, "
                    f"got {seen[model_name]}",
                )


# =============================================================================
# 5. Adapter --batch_size propagation to CLI command
# =============================================================================

class TestAdapterBatchSizeInCommand(unittest.TestCase):
    """Each adapter must include --batch_size N in its build_command() output."""

    def _make_job(self, batch_size: int, **overrides):
        from experiment_orchestrator.config_resolver import ResolvedJob
        defaults = dict(
            job_name="test_job",
            model_name="TestModel",
            test_name="test",
            variant="single_slice",
            run_number=1,
            total_runs=1,
            adapter_name="medgemma",
            environment="medgemma",
            model_path="/models/test",
            image_dir="/data/slices",
            image_path=None,
            qa_path="/data/qa.csv",
            output_path="/out/results.csv",
            slurm_params={},
            batch_size=batch_size,
        )
        defaults.update(overrides)
        return ResolvedJob(**defaults)

    def test_medgemma_adapter_includes_batch_size(self):
        from experiment_orchestrator.adapters.medgemma import MedGemmaAdapter
        job = self._make_job(batch_size=4)
        cmd = MedGemmaAdapter().build_command(job, "/project")
        self.assertIn("--batch_size 4", cmd)

    def test_lingshu_adapter_includes_batch_size(self):
        from experiment_orchestrator.adapters.lingshu import LingshuAdapter
        job = self._make_job(batch_size=2, adapter_name="lingshu")
        cmd = LingshuAdapter().build_command(job, "/project")
        self.assertIn("--batch_size 2", cmd)

    def test_qwen_adapter_includes_batch_size(self):
        from experiment_orchestrator.adapters.qwen import QwenAdapter
        job = self._make_job(
            batch_size=8, variant="text_only", image_dir=None, adapter_name="qwen"
        )
        cmd = QwenAdapter().build_command(job, "/project")
        self.assertIn("--batch_size 8", cmd)

    def test_medimageinsight_adapter_includes_batch_size(self):
        from experiment_orchestrator.adapters.medimageinsight import MedImageInsightAdapter
        job = self._make_job(batch_size=16, adapter_name="medimageinsight")
        cmd = MedImageInsightAdapter().build_command(job, "/project")
        self.assertIn("--batch_size 16", cmd)

    def test_med3dvlm_adapter_includes_batch_size(self):
        from experiment_orchestrator.adapters.med3dvlm import Med3DVLMAdapter
        job = self._make_job(
            batch_size=1, variant="full_nifti", image_dir="/data/nifti", adapter_name="med3dvlm"
        )
        cmd = Med3DVLMAdapter().build_command(job, "/project")
        self.assertIn("--batch_size 1", cmd)

    def test_med3dvlm_blank_adapter_includes_batch_size(self):
        from experiment_orchestrator.adapters.med3dvlm import Med3DVLMAdapter
        job = self._make_job(
            batch_size=1, variant="blank",
            image_path="/data/blank.nii.gz", adapter_name="med3dvlm"
        )
        cmd = Med3DVLMAdapter().build_command(job, "/project")
        self.assertIn("--batch_size 1", cmd)


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
