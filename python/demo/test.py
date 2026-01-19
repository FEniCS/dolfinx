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


def imports_petsc4py(f):
    """Check if a file imports petsc4py."""
    with open(f, encoding="utf-8") as file:
        read_data = file.read()
        return "petsc4py" in read_data or ".petsc" in read_data


# Get directory of this file
path = pathlib.Path(__file__).resolve().parent

# Build list of demo programs
demo_files = list(path.glob("**/*.py"))
if importlib.util.find_spec("petsc4py") is not None:
    demos = [(f.parent, f.name) for f in demo_files]
else:
    demos = [(f.parent, f.name) for f in demo_files if not imports_petsc4py(f)]


@pytest.mark.serial
@pytest.mark.parametrize("path,name", demos)
def test_demos(path, name):
    """Test demo scripts in serial."""
    ret = subprocess.run([sys.executable, name], cwd=str(path), check=True)
    assert ret.returncode == 0


@pytest.mark.mpi
@pytest.mark.parametrize("path,name", demos)
def test_demos_mpi(num_proc, mpiexec, path, name):
    """Test demo scripts in parallel using MPI."""
    cmd = [mpiexec, "-np", str(num_proc), sys.executable, name]
    print(cmd)
    ret = subprocess.run(cmd, cwd=str(path), check=True)
    assert ret.returncode == 0
