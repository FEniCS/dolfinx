# Copyright (C) 2016-2025 Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Demo test helpers."""

import importlib.util
import pathlib
import subprocess
import sys

import pytest

# Bound on a single demo's run time, so a deadlock (e.g. an MPI
# collective mismatch) fails fast with a clear TimeoutExpired instead
# of hanging until CI's own multi-hour job timeout.
DEMO_TIMEOUT_S = 300


# Demos importing one of these modules are skipped when the module is
# not installed, rather than failing.
OPTIONAL_DEMO_MODULES = ["petsc4py", "gmsh", "pyvista"]


def imports_module(f, module):
    """Check if a file imports a given (optional) module."""
    with open(f, encoding="utf-8") as file:
        read_data = file.read()
    if module == "petsc4py":
        return "petsc4py" in read_data or ".petsc" in read_data
    return f"import {module}" in read_data


# Get directory of this file
path = pathlib.Path(__file__).resolve().parent

# Build list of demo programs, skipping ones that import an optional
# module not installed in this environment.
demo_files = list(path.glob("**/*.py"))
missing_modules = [m for m in OPTIONAL_DEMO_MODULES if importlib.util.find_spec(m) is None]
demos = [
    (f.parent, f.name) for f in demo_files if not any(imports_module(f, m) for m in missing_modules)
]


@pytest.mark.serial
@pytest.mark.parametrize("path,name", demos)
def test_demos(path, name):
    """Test demo scripts in serial."""
    ret = subprocess.run([sys.executable, name], cwd=str(path), check=True, timeout=DEMO_TIMEOUT_S)
    assert ret.returncode == 0


@pytest.mark.mpi
@pytest.mark.parametrize("path,name", demos)
def test_demos_mpi(num_proc, mpiexec, path, name):
    """Test demo scripts in parallel using MPI."""
    cmd = [mpiexec, "-np", str(num_proc), sys.executable, name]
    print(cmd)
    ret = subprocess.run(cmd, cwd=str(path), check=True, timeout=DEMO_TIMEOUT_S)
    assert ret.returncode == 0
