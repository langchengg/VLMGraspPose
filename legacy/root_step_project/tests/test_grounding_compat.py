import unittest
from unittest import mock

from src import grounding


class FlorenceTransformersCompatTests(unittest.TestCase):
    def test_rejects_transformers_v5_for_florence(self):
        with self.assertRaises(RuntimeError) as ctx:
            grounding.ensure_florence_transformers_compat("5.5.4")
        self.assertIn("transformers<5", str(ctx.exception))

    def test_allows_transformers_v4_for_florence(self):
        grounding.ensure_florence_transformers_compat("4.57.0")

    def test_uses_eager_attention_for_florence_remote_code(self):
        kwargs = grounding.florence_model_load_kwargs("float32")
        self.assertEqual(kwargs["attn_implementation"], "eager")
        self.assertTrue(kwargs["trust_remote_code"])

    def test_disables_generation_cache_for_florence(self):
        kwargs = grounding.florence_generation_kwargs()
        self.assertFalse(kwargs["use_cache"])

    def test_prepares_local_florence_environment(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            grounding.prepare_florence_environment()
            import os
            self.assertEqual(os.environ["USE_TF"], "0")
            self.assertIn(".hf_cache", os.environ["HF_HOME"])


if __name__ == "__main__":
    unittest.main()
