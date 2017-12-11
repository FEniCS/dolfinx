import subprocess
import sys
import pytest

def test_demos():
    ret = subprocess.run([sys.executable, "./documented/poisson/demo_poisson.py"],
                         check=True)
    #assert ret
    print("---------------------")
    print(ret.returncode)
