import os
import pathlib
import subprocess
import sys
import pytest


# Build list of demo programs
p = pathlib.Path('documented')
demo_files = list(p.glob('**/*.py'))
demos = []
for f in demo_files:
    demos.append((f.parent, f.name))


@pytest.mark.parametrize("path,name", demos)
def test_demos(name, path):
    ret = subprocess.run([sys.executable, name],
                         cwd=path,
                         env={**os.environ, 'MPLBACKEND': 'agg'},
                         check=True)
