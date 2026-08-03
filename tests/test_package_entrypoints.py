import importlib
from pathlib import Path
import tomllib
import unittest


class PackageEntrypointTests(unittest.TestCase):
    def test_every_console_script_resolves_to_a_callable(self) -> None:
        root = Path(__file__).resolve().parents[1]
        metadata = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
        scripts = metadata["project"]["scripts"]

        for name, target in scripts.items():
            with self.subTest(name=name):
                module_name, attribute = target.split(":", 1)
                value = getattr(importlib.import_module(module_name), attribute)
                self.assertTrue(callable(value), target)


if __name__ == "__main__":
    unittest.main()
