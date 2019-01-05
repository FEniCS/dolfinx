# -*- coding: utf-8 -*-
# Copyright (C) 2016-2016 Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import os
import pathlib
import subprocess
import sys

import pytest

# Get directory of this file
dir_path = pathlib.Path(__file__).resolve().parent

# Build list of demo programs
demos = []
for subdir in ['documented', 'undocumented']:
    p = pathlib.Path(dir_path, subdir)
    demo_files = list(p.glob('**/*.py'))
    for f in demo_files:
        demos.append((f.parent, f.name))


@pytest.mark.serial
@pytest.mark.parametrize("path,name", demos)
def test_demos(path, name):
    ret = subprocess.run([sys.executable, name],
                         cwd=str(path),
                         env={**os.environ, 'MPLBACKEND': 'agg', 'DOLFIN_TEST': '1'},
                         check=True)
    assert ret.returncode == 0


@pytest.mark.mpi
@pytest.mark.parametrize("path,name", demos)
def test_demos_mpi(num_proc, mpiexec, path, name):
    cmd = [mpiexec, "-np", str(num_proc), sys.executable, name]
    ret = subprocess.run(cmd,
                         cwd=str(path),
                         env={**os.environ, 'MPLBACKEND': 'agg', 'DOLFIN_TEST': '1'},
                         check=True)
    assert ret.returncode == 0
