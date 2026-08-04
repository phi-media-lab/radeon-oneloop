import unittest

from gaussian.canonicalize_vksplat_ply import inverse_similarity


class ObservedCanonicalAuditTests(unittest.TestCase):
    def test_identity_transform_is_valid_audit_transform(self):
        import numpy as np

        scale, rotation, translation = inverse_similarity(np.eye(4))
        self.assertEqual(scale, 1.0)
        np.testing.assert_array_equal(rotation, np.eye(3))
        np.testing.assert_array_equal(translation, np.zeros(3))


if __name__ == "__main__":
    unittest.main()
