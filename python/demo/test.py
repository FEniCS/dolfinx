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

@pytest.mark.parametrize("path,name", demos)
def test_demos(path, name):

    # Run in serial
    ret = subprocess.run([sys.executable, name],
                         cwd=str(path),
                         env={**os.environ, 'MPLBACKEND': 'agg', 'DOLFIN_TEST': '1'},
                         check=True)


@pytest.mark.parametrize("path,name", demos)
def test_demos_mpi(path, name):

    # Run in parallel
    cmd = ["mpirun", "-np", "3", sys.executable, name]
    ret = subprocess.run(cmd,
                         cwd=str(path),
                         env={**os.environ, 'MPLBACKEND': 'agg', 'DOLFIN_TEST': '1'},
                         check=True)
