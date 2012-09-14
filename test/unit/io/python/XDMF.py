"""Unit tests for the XDMF io library"""

# Copyright (C) 2012 Garth N. Wells
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
# First added:  2012-09-14
# Last changed:

import unittest
from dolfin import *

class XDMF_Mesh_Output(unittest.TestCase):
    """Test output of Meshes to XDMF files"""

    def test_save_1d_mesh(self):
        if MPI.num_processes() == 1:
            mesh = UnitInterval(32)
            File("mesh.xdmf") << mesh

    def test_save_2d_mesh(self):
        mesh = UnitSquare(32, 32)
        File("mesh.xdmf") << mesh

    def test_save_3d_mesh(self):
        mesh = UnitCube(8, 8, 8)
        File("mesh.xdmf") << mesh


class XDMF_Point_Function_Output(unittest.TestCase):
    """Test output of point-based Functions to XDMF files"""

    def test_save_1d_scalar(self):
        if MPI.num_processes() == 1:
            mesh = UnitInterval(32)
            u = Function(FunctionSpace(mesh, "Lagrange", 2))
            u.vector()[:] = 1.0
            File("u.xdmf") << u

    def test_save_2d_scalar(self):
        mesh = UnitSquare(16, 16)
        u = Function(FunctionSpace(mesh, "Lagrange", 2))
        u.vector()[:] = 1.0
        File("u.xdmf") << u

    def test_save_3d_scalar(self):
        mesh = UnitCube(8, 8, 8)
        u = Function(FunctionSpace(mesh, "Lagrange", 2))
        u.vector()[:] = 1.0
        File("u.xdmf") << u

    # FFC fails for vector spaces in 1D
    #def test_save_1d_vector(self):
    #    if MPI.num_processes() == 1:
    #        mesh = UnitInterval(32)
    #        u = Function(VectorFunctionSpace(mesh, "Lagrange", 2))
    #        u.vector()[:] = 1.0
    #        File("u.xdmf") << u

    def test_save_2d_vector(self):
        mesh = UnitSquare(16, 16)
        u = Function(VectorFunctionSpace(mesh, "Lagrange", 2))
        u.vector()[:] = 1.0
        File("u.xdmf") << u

    def test_save_3d_vector(self):
        mesh = UnitCube(8, 8, 8)
        u = Function(VectorFunctionSpace(mesh, "Lagrange", 2))
        u.vector()[:] = 1.0
        File("u.xdmf") << u

    # FFC fails for tensor spaces in 1D
    #def test_save_1d_tensor(self):
    #    if MPI.num_processes() == 1:
    #        mesh = UnitInterval(32)
    #        u = Function(TensorFunctionSpace(mesh, "Lagrange", 2))
    #        u.vector()[:] = 1.0
    #        File("u.xdmf") << u

    def test_save_2d_tensor(self):
        mesh = UnitSquare(16, 16)
        u = Function(TensorFunctionSpace(mesh, "Lagrange", 2))
        u.vector()[:] = 1.0
        File("u.xdmf") << u

    def test_save_3d_tensor(self):
        mesh = UnitCube(8, 8, 8)
        u = Function(TensorFunctionSpace(mesh, "Lagrange", 2))
        u.vector()[:] = 1.0
        File("u.xdmf") << u

if __name__ == "__main__":
    unittest.main()
