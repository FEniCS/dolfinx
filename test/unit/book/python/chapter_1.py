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
# Last changed: 2011-10-20

import unittest
import inspect, runpy, os
from dolfin import *

def run_test(path):
    "Run test script implied by name of calling function, neat trick..."
    script_name = inspect.stack()[1][3].split("test_")[1] + ".py"
    file_path = os.path.join(*(["chapter_1_files"] + path + [script_name]))
    runpy.run_path(file_path)

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
        run_test(["stationary", "poisson"])

    def test_d3_p2D(self):
        run_test(["stationary", "poisson"])

    def test_d6_p2D(self):
        run_test(["stationary", "poisson"])

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

    def test_vcp2D(self):
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
        run_test(["stationary", "nonlinear_poisson"])

    def test_picard_np(self):
        run_test(["stationary", "nonlinear_poisson"])

    def test_vp2_np(self):
        run_test(["stationary", "nonlinear_poisson"])

    def test_alg_newton_np(self):
        run_test(["stationary", "nonlinear_poisson"])

    def test_vp1_np(self):
        run_test(["stationary", "nonlinear_poisson"])

class TestDiffusion(unittest.TestCase):

    def test_d1_d2D(self):
        run_test(["transient", "diffusion"])

    def test_demo_sin_daD(self):
        run_test(["transient", "diffusion"])

    def test_d2_d2D(self):
        run_test(["transient", "diffusion"])

    def test_sin_daD(self):
        run_test(["transient", "diffusion"])

if __name__ == "__main__":
    print ""
    print "Testing the FEniCS Book, Chapter 1"
    print "----------------------------------"
    unittest.main()
