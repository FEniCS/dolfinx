# -*- coding: utf-8 -*-
# Copyright (C) 2021 Garth N. Wells and Jørgen S. Dokken
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import os
import pathlib
import subprocess
import sys

import pytest

# Get directory of this file
path = pathlib.Path(__file__).resolve().parent

# List of demo programs using pyvista
pyvista_demos = ["demo_poisson.py",
                 "demo_cahn-hilliard.py",
                 "demo_pyvista.py"]

# Build list of demo programs
demos = []
demo_files = list(path.glob('**/*.py'))
for f in demo_files:
    if f.name in pyvista_demos:
        demos.append((f.parent, f.name))


@pytest.mark.serial
@pytest.mark.parametrize("path,name", demos)
def test_pyvista_demos(path, name):
    ret = subprocess.run([sys.executable, name],
                         cwd=str(path),
                         env={**os.environ, 'MPLBACKEND': 'agg'},
                         check=True)
    assert ret.returncode == 0


@pytest.mark.mpi
@pytest.mark.parametrize("path,name", demos)
def test_pyvista_demos_mpi(num_proc, mpiexec, path, name):
    cmd = [mpiexec, "-np", str(num_proc), sys.executable, name]
    print(cmd)
    ret = subprocess.run(cmd,
                         cwd=str(path),
                         env={**os.environ, 'MPLBACKEND': 'agg'},
                         check=True)
    assert ret.returncode == 0
