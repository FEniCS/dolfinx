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


# FIXME: fix cases that break pattern
# Remove 'tensor-weighted-poisson'
demos = [d for d in demos if d[0].stem != 'tensor-weighted-poisson']

# Testing
#demos = [d for d in demos if d[0].stem == 'stokes-taylor-hood']
#print("------------------------------")
#print(demos)


@pytest.mark.parametrize("path,name", demos)
def test_demos(mpiexec, num_proc, path, name):

    if mpiexec is None:
        # Run in serial
        ret = subprocess.run([sys.executable, name],
                             cwd=str(path),
                             env={**os.environ, 'MPLBACKEND': 'agg'},
                             check=True)
    else:
        # Run with MPI

        # FIXME: non-MPI demos should exit gracefully
        # Demos that don't work in parallel
        broken = ["demo_subdomains.py",
                  "demo_auto-adaptive-poisson.py",
                  "demo_nonmatching-interpolation.py"
                  'demo_adaptive-poisson,py',
                  'demo_auto-adaptive-navier-stokes.py',
                  'demo_eval.py',
                  'demo_extrapolation.py',
                  'demo_nonmatching-interpolation.py',
                  'demo_nonmatching-projection.py',
                  'demo_poisson-disc.py',
                  'demo_smoothing.py',
                  'demo_subdomains.py',
                  'demo_submesh.py',
                  'demo_time-series.py',
                  'demo_poisson1D-in-2D.py',
                  'demo_coordinates.py',
        ]
        if name in broken:
            return

        assert int(num_proc) > 0 and int(num_proc) < 32
        cmd = mpiexec + " -np " + str(num_proc) + " " + sys.executable + " " + name
        ret = subprocess.run(cmd,
                             cwd=str(path),
                             shell=True,
                             #stdin=subprocess.DEVNULL,
                             env={**os.environ, 'MPLBACKEND': 'agg'},
                             check=True
        )
