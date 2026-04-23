"""Setup script for jax_rebo2.

The vendored ``interpax`` submodule (under ``./interpax``) is installed as a
dependency at ``pip install -e .`` time via a PEP 508 direct file:// URL.
This requires that the git submodule has been initialized, e.g.::

    git submodule update --init --recursive

The submodule's own ``requirements.txt`` pins stale upper bounds on ``jax``
and ``numpy`` (``jax < 0.7``, ``numpy < 2.3``) that conflict with the
known-working environment for ``jax_rebo2`` (``jax 0.8.2``, ``numpy 2.1``).
To let a single ``pip install -e .`` succeed, the submodule is copied to a
build-side staging directory (``build/interpax_relaxed``) with those upper
bounds dropped, and that staging copy is what ``pip`` actually installs. The
submodule working tree itself is never modified.

Optional extras:
- ``lammps``      : installs the LAMMPS Python bindings (needed for
                    ``jax_rebo2.testing`` and ``jax_rebo2.utils.pair_force``).
- ``dataset``     : installs ``dpdata`` (needed for
                    ``jax_rebo2.build_dataset.make_original_pickle``).
- ``all``         : all of the above.
"""

import re
import shutil
from pathlib import Path

from setuptools import find_packages, setup

HERE = Path(__file__).parent.resolve()
INTERPAX_SRC = HERE / "interpax"
INTERPAX_STAGE = HERE / "build" / "interpax_relaxed"


def _prepare_interpax_stage() -> Path:
    """Copy the interpax submodule into a staging dir and drop stale upper
    bounds on ``jax`` / ``numpy`` in its ``requirements.txt`` so our pinned
    ``jax>=0.8`` / ``numpy>=2.1`` can coexist with it."""
    if not (INTERPAX_SRC / "setup.py").exists() and not (
        INTERPAX_SRC / "pyproject.toml"
    ).exists():
        raise RuntimeError(
            "The bundled 'interpax' submodule is missing or empty.\n"
            "Run: git submodule update --init --recursive\n"
            f"Expected to find a Python package at: {INTERPAX_SRC}"
        )

    if INTERPAX_STAGE.exists():
        shutil.rmtree(INTERPAX_STAGE)
    INTERPAX_STAGE.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        INTERPAX_SRC,
        INTERPAX_STAGE,
        ignore=shutil.ignore_patterns(".git", ".github", "docs", "tests"),
    )

    req_file = INTERPAX_STAGE / "requirements.txt"
    if req_file.exists():
        relaxed_lines = []
        for line in req_file.read_text().splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                relaxed_lines.append(line)
                continue
            # Match "name <specifiers>"; keep only lower bounds.
            m = re.match(r"^\s*([A-Za-z0-9_.\-]+)\s*(.*)$", stripped)
            if not m:
                relaxed_lines.append(line)
                continue
            name, specs = m.group(1), m.group(2)
            keep = [
                s.strip()
                for s in specs.split(",")
                if s.strip().startswith(">=") or s.strip().startswith(">")
            ]
            relaxed_lines.append(f"{name} {', '.join(keep)}".strip())
        req_file.write_text("\n".join(relaxed_lines) + "\n")

    return INTERPAX_STAGE


INTERPAX_STAGED = _prepare_interpax_stage()
INTERPAX_REQ = f"interpax @ file://{INTERPAX_STAGED.as_posix()}"

readme_path = HERE / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Version floors taken from a known-working `jaxmd` conda environment.
install_requires = [
    INTERPAX_REQ,
    "jax>=0.8.2",
    "jaxlib>=0.8.2",
    "jax-md>=0.2.8",
    "numpy>=2.1,<3",
    "optax>=0.2.4",
    "chex>=0.1.88",
    "flax>=0.10.3",
    "ml-dtypes>=0.5.1",
    "matplotlib>=3.10",
    "pandas>=2.2",
    "h5py>=3.12",
    "tqdm>=4.67",
]

extras_require = {
    "lammps": ["lammps>=2025.7.22"],
    "dataset": ["dpdata>=0.2.23"],
}
extras_require["all"] = sorted({pkg for deps in extras_require.values() for pkg in deps})

setup(
    name="jax_rebo2",
    version="0.1.0",
    description="JAX implementation of the REBO2 interatomic potential.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["interpax", "interpax.*", "scripts", "scripts.*"]),
    python_requires=">=3.12",
    install_requires=install_requires,
    extras_require=extras_require,
)
