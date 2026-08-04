import unittest

from gaussian.vista4d_inference_cfg import install_cfg_scale_override


class Vista4DCfgOverrideTests(unittest.TestCase):
    def test_override_supplies_default_but_preserves_explicit_value(self):
        class FakePipeline:
            def __call__(self, **kwargs):
                return kwargs["cfg_scale"]

        install_cfg_scale_override(FakePipeline, 3.0)
        pipeline = FakePipeline()
        self.assertEqual(pipeline(), 3.0)
        self.assertEqual(pipeline(cfg_scale=2.0), 2.0)

    def test_override_rejects_implausible_scale(self):
        class FakePipeline:
            def __call__(self, **kwargs):
                return kwargs

        with self.assertRaisesRegex(ValueError, r"\[1, 10\]"):
            install_cfg_scale_override(FakePipeline, 0.5)


if __name__ == "__main__":
    unittest.main()
