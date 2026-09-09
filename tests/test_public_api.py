from pathlib import Path
import subprocess
import sys

import fiatlux
import fiatlux.optics as optics


ROOT = Path(__file__).parents[1]


def test_top_level_public_exports_are_importable():
    for name in fiatlux.__all__:
        assert getattr(fiatlux, name) is not None


def test_optics_public_exports_are_importable():
    for name in optics.__all__:
        assert getattr(optics, name) is not None


def test_core_import_does_not_load_optional_dependencies():
    script = """
import sys
import fiatlux

optional_modules = {"astropy", "matplotlib", "scipy", "torchvision"}
loaded = optional_modules.intersection(sys.modules)
assert not loaded, f"optional dependencies loaded by import fiatlux: {sorted(loaded)}"
"""
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        text=True,
        capture_output=True,
        cwd=ROOT,
    )
