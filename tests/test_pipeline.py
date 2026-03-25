import json
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import METRICS_PATH, PROCESSED_DATA_PATH, SUMMARY_PATH
from src.pipeline import run_pipeline


class PipelineTestCase(unittest.TestCase):
    def test_pipeline_runs_and_generates_outputs(self):
        summary = run_pipeline(refresh_download=False)
        self.assertTrue(PROCESSED_DATA_PATH.exists())
        self.assertTrue(METRICS_PATH.exists())
        self.assertTrue(SUMMARY_PATH.exists())
        self.assertIn(summary["best_model"], {"logistic_regression", "linear_svc", "multinomial_nb"})
        self.assertGreater(summary["best_macro_f1"], 0.55)
        saved = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
        self.assertEqual(saved["rows"], summary["rows"])


if __name__ == "__main__":
    unittest.main()
