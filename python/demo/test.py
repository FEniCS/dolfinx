# Copyright (C) 2016-2016 Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import pathlib
import subprocess
import sys

import pytest

# Get directory of this file
path = pathlib.Path(__file__).resolve().parent

# Build list of demo programs
demos = []
demo_files = list(path.glob("**/*.py"))
for f in demo_files:
    demos.append((f.parent, f.name))


@pytest.mark.serial
@pytest.mark.parametrize("path,name", demos)
def test_demos(path, name):
    if "pyamg" in name:
        pytest.xfail("pyamg needs updating for latest numpy and scipy")
    ret = subprocess.run([sys.executable, name], cwd=str(path), check=True)
    assert ret.returncode == 0


@pytest.mark.mpi
@pytest.mark.parametrize("path,name", demos)
def test_demos_mpi(num_proc, mpiexec, path, name):
    cmd = [mpiexec, "-np", str(num_proc), sys.executable, name]
    print(cmd)
    ret = subprocess.run(cmd, cwd=str(path), check=True)
    assert ret.returncode == 0
