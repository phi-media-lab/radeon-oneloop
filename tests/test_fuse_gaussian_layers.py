from __future__ import annotations

from gaussian.fuse_gaussian_layers import DONE_SCHEMA, SCHEMA


def test_layered_fusion_schemas_are_distinct_and_versioned() -> None:
    assert SCHEMA.endswith(".v1")
    assert DONE_SCHEMA.endswith(".v1")
    assert SCHEMA != DONE_SCHEMA
