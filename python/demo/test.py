import subprocess
import sys
import pytest

# Build list of demos

@pytest.mark.parametrize("name,path",
                         [
                             ("demo_biharmonic.py", "./documented/biharmonic/"),
                             ("demo_poisson.py", "./documented/poisson/"),
                         ])
def test_demos(name, path):
    #ret = subprocess.run([sys.executable, "demo_poisson.py"],
    #                     shell=True, check=True, cwd="./documented/poisson/")
    ret = subprocess.run([sys.executable, name],
                         cwd=path,
                         #env={"DOLFIN_NOPLOT": "0",
                         #     "PYTHONPATH": "/home/garth/code/fenics/dev/dolfin.d/dolfin/python/build/lib.linux-x86_64-3.6/"},
                         check=True)
    #assert ret
    #print("---------------------")
    #print(ret.returncode)
