"""Unit tests for assembly"""

# Copyright (C) 2011 Garth N. Wells
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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
#
# First added:  2011-03-12
# Last changed: 2011-03-12

import unittest
import numpy
from dolfin import *

class Assembly(unittest.TestCase):

    def test_functional_assembly(self):

        mesh = UnitSquare(24, 24)

        # This is a hack to get around a DOLFIN bug
        if MPI.num_processes() > 1:
            cpp.MeshPartitioning.number_entities(mesh, mesh.topology().dim() - 1);

        f = Constant(1.0)
        M = f*dx
        parameters["num_threads"] = 0
        self.assertAlmostEqual(assemble(M, mesh=mesh), 1.0)

        M = f*ds
        parameters["num_threads"] = 0
        self.assertAlmostEqual(assemble(M, mesh=mesh), 4.0)

    def test_colored_cell_assembly(self):

        # Coloring and renumbering not supported in parallel
        if MPI.num_processes() != 1:
            return

        # Create mesh, then color and renumber
        old_mesh = UnitCube(4, 4, 4)
        old_mesh.color("vertex")
        mesh = old_mesh.renumber_by_color()

        V = VectorFunctionSpace(mesh, "DG", 1)
        v = TestFunction(V)
        u = TrialFunction(V)
        f = Constant((10, 20, 30))
        def epsilon(v):
            return 0.5*(grad(v) + grad(v).T)
        a = inner(epsilon(v), epsilon(u))*dx
        L = inner(v, f)*dx

        A_frobenius_norm =  4.3969686527582512
        b_l2_norm = 0.95470326978246278

        # Assemble A and b separately
        parameters["num_threads"] = 0
        self.assertAlmostEqual(assemble(a).norm("frobenius"), A_frobenius_norm, 10)
        self.assertAlmostEqual(assemble(L).norm("l2"), b_l2_norm, 10)

        # Assemble A and b separately (multi-threaded)
        parameters["num_threads"] = 4
        self.assertAlmostEqual(assemble(a).norm("frobenius"), A_frobenius_norm, 10)
        self.assertAlmostEqual(assemble(L).norm("l2"), b_l2_norm, 10)

if __name__ == "__main__":
    print ""
    print "Testing basic DOLFIN assembly operations"
    print "------------------------------------------------"
    unittest.main()
