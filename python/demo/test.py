import os
import pathlib
import subprocess
import sys
import pytest


# Get directory of this file
dir_path = pathlib.Path(__file__).resolve().parent

# Build list of demo programs
demos = []
for subdir in ['documented']:
    p = pathlib.Path(dir_path, subdir)
    demo_files = list(p.glob('**/*.py'))
    for f in demo_files:
        demos.append((f.parent, f.name))

# FIXME: remove cases that break pattern
# Remove 'tensor-weighted-poisson'
demos = [d for d in demos if d[0].stem != 'tensor-weighted-poisson']

# Testing
#demos = [d for d in demos if d[0].stem == 'stokes-taylor-hood']
#print("------------------------------")
#print(demos)

@pytest.mark.parametrize("path,name", demos)
def test_demos(name, path):
    ret = subprocess.run([sys.executable, name],
                         cwd=path,
                         env={**os.environ, 'MPLBACKEND': 'agg'},
                         check=True)
