"""
Unit tests for Chapter 1 (A FEniCS tutorial).
"""

# Copyright (C) 2011 Hans Petter Langtangen and Anders Logg
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
#
# First added:  2011-10-20
# Last changed: 2011-11-08

import unittest
import inspect, os, sys
from dolfin import *

# Try importing runpy, added in Python 2.7
try:
    from runpy import run_path as runpy_run_path
    has_run_path = True
except:
    has_run_path = False

def run_path(path, args):
    "Replacement for runpy.run_path when it doesn't exist"

    if has_run_path:
        sys.argv = ["foo"] + [str(arg) for arg in args]
        runpy_run_path(path)
    else:
        status = os.system("python " + path + " " + \
                           " ".join(str(arg) for arg in args))
        if not status == 0:
            raise RuntimeError, "Python script failed"

def skip_in_parallel():
    "Skip test in parallel"
    if MPI.num_processes() > 1:
        print "FIXME: This unit test does not work in parallel, skipping"
        return True
    return False

def run_test(path, args=[]):
    "Run test script implied by name of calling function, neat trick..."

    # Skip tests in parallel for now
    if skip_in_parallel(): return

    # Figure out name of script to be run
    script_name = inspect.stack()[1][3].split("test_")[1] + ".py"
    file_path = os.path.join(*(["chapter_1_files"] + path + [script_name]))

    # Print a message
    print
    print "Running tutorial example %s" % file_path
    print "-------------------------------------------------------------------------"

    # Remember default DOLFIN parameters
    dolfin_parameters = {}
    dolfin_parameters.update(parameters)

    # Run script with default parameters
    run_path(file_path, args)

    # Set parameters from file (as they were at the time of book release)
    file = File(os.path.join("chapter_1_files", "dolfin_parameters.xml"))
    file >> parameters

    # Run script again with book parameters
    run_path(file_path, args)

    # Reset parameters
    parameters.update(dolfin_parameters)

class TestPoisson(unittest.TestCase):

    def test_dn3_p2D(self):
        run_test(["stationary", "poisson"])

    def test_dn3_p2D(self):
        run_test(["stationary", "poisson"])

    def test_dnr_p2D(self):
        run_test(["stationary", "poisson"])

    def test_d5_p2D(self):
        run_test(["stationary", "poisson"])

    def test_d1_p2D(self):
        run_test(["stationary", "poisson"])

    def test_paD(self):
        run_test(["stationary", "poisson"], [8, 8])

    def test_d3_p2D(self):
        run_test(["stationary", "poisson"], [1])

    def test_d6_p2D(self):
        run_test(["stationary", "poisson"], [1])

    def test_dn2_p2D(self):
        run_test(["stationary", "poisson"])

    def test_d2_p2D(self):
        run_test(["stationary", "poisson"])

    def test_mat2x_p2D(self):
        run_test(["stationary", "poisson"])

    def test_dn1_p2D(self):
        run_test(["stationary", "poisson"])

    def test_dn4_p2D(self):
        run_test(["stationary", "poisson"])

    def disabled_test_vcp2D(self):
        # Disabled since it depends on scitools
        run_test(["stationary", "poisson"])

    def test_d4_p2D(self):
        run_test(["stationary", "poisson"])

    def test_mat2_p2D(self):
        run_test(["stationary", "poisson"])

    def test_membrane1v(self):
        run_test(["stationary", "poisson"])

    def test_membrane1(self):
        run_test(["stationary", "poisson"])

class TestNonlinearPoisson(unittest.TestCase):

    def test_pde_newton_np(self):
        run_test(["stationary", "nonlinear_poisson"], [1, 8, 8])

    def test_picard_np(self):
        run_test(["stationary", "nonlinear_poisson"], [1, 8, 8])

    def test_vp1_np(self):
        run_test(["stationary", "nonlinear_poisson"], ["a", "g", 1, 8, 8])

    def test_vp2_np(self):
        run_test(["stationary", "nonlinear_poisson"], ["a", "g", 1, 8, 8])

    def test_alg_newton_np(self):
        run_test(["stationary", "nonlinear_poisson"], [1, 8, 8])

class TestDiffusion(unittest.TestCase):

    def test_d1_d2D(self):
        run_test(["transient", "diffusion"])

    def test_d2_d2D(self):
        run_test(["transient", "diffusion"])

    def disabled_test_sin_daD(self):
        # Disabled since it depends on scitools
        run_test(["transient", "diffusion"], [1, 1.5, 4, 40])

if __name__ == "__main__":
    print ""
    print "Testing the FEniCS Book, Chapter 1"
    print "----------------------------------"
    unittest.main()
