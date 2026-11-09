"""
Run every Python example of the README and of the examples folder, so that a change of the API or of a default
cannot leave the documentation broken. Examples that need an optional package are skipped when it is missing.
"""

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
OPTIONAL = ("fem2d", "struct_core", "struct_load")


def _readme_blocks():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)```", text, flags=re.S)
    return [pytest.param(code, id=f"README-block-{i}") for i, code in enumerate(blocks, 1)]


def _example_scripts():
    scripts = sorted(p for p in (ROOT / "examples").rglob("*.py")
                     if p.name not in ("__init__.py", "run_all.py") and "trash" not in p.parts)
    return [pytest.param(p, id=str(p.relative_to(ROOT).as_posix())) for p in scripts]


def _skip_if_optional_missing(code):
    """Skip when the code imports an optional package at top level (an indented import is guarded by the code)."""
    for name in OPTIONAL:
        if re.search(rf"^(from|import)\s+{name}\b", code, flags=re.M) and importlib.util.find_spec(name) is None:
            pytest.skip(f"needs the optional package {name}")


def _run(args, cwd):
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONIOENCODING="utf-8")
    result = subprocess.run([sys.executable, *args], cwd=cwd, env=env, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, f"exit code {result.returncode}\n{result.stderr[-3000:]}"


@pytest.mark.parametrize("code", _readme_blocks())
def test_readme_example_runs(code, tmp_path):
    _skip_if_optional_missing(code)
    script = tmp_path / "example.py"
    script.write_text(code, encoding="utf-8")
    _run([str(script)], cwd=tmp_path)


@pytest.mark.parametrize("script", _example_scripts())
def test_example_script_runs(script):
    _skip_if_optional_missing(script.read_text(encoding="utf-8"))
    _run([script.name], cwd=script.parent)
