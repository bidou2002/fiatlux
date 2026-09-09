from pathlib import Path
import tomllib

import fiatlux


ROOT = Path(__file__).parents[1]


def test_runtime_version_comes_from_packaging_source():
    namespace = {}
    exec((ROOT / "fiatlux" / "_version.py").read_text(), namespace)
    assert fiatlux.__version__ == namespace["__version__"]


def test_pyproject_declares_supported_python_license_and_dependencies():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
    project = metadata["project"]

    assert project["requires-python"] == ">=3.10"
    assert project["license"] == "MIT"
    assert project["dependencies"] == ["numpy>=1.23.5", "torch>=2.0"]
    assert {"fits", "plot", "tutorials", "dev", "all"} <= set(
        project["optional-dependencies"]
    )


def test_deprecated_packaging_files_are_removed():
    assert not (ROOT / "setup.py").exists()
    assert not (ROOT / "fiatlux" / "requirements.txt").exists()
