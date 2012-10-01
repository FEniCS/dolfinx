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

    # Disabled because 1D not supported yet
    def xtest_save_1d_mesh(self):
        if MPI.num_processes() == 1:
            mesh = UnitInterval(32)
            File("mesh.xdmf") << mesh

    def test_save_2d_mesh(self):
        mesh = UnitSquare(32, 32)
        File("mesh_2D.xdmf") << mesh
        print "Done", MPI.process_number()

    def xtest_save_3d_mesh(self):
        mesh = UnitCube(8, 8, 8)
        File("mesh_3D.xdmf") << mesh


class XDMF_Vertex_Function_Output(unittest.TestCase):
    """Test output of vertex-based Functions to XDMF files"""

    # Disabled because 1D not supported yet
    def xtest_save_1d_scalar(self):
        if MPI.num_processes() == 1:
            mesh = UnitInterval(32)
            u = Function(FunctionSpace(mesh, "Lagrange", 2))
            u.vector()[:] = 1.0
            File("u.xdmf") << u

    def xtest_save_2d_scalar(self):
        mesh = UnitSquare(16, 16)
        u = Function(FunctionSpace(mesh, "Lagrange", 2))
        u.vector()[:] = 1.0
        File("u.xdmf") << u

    def xtest_save_3d_scalar(self):
        mesh = UnitCube(8, 8, 8)
        u = Function(FunctionSpace(mesh, "Lagrange", 2))
        u.vector()[:] = 1.0
        File("u.xdmf") << u

    def xtest_save_2d_vector(self):
        mesh = UnitSquare(16, 16)
        u = Function(VectorFunctionSpace(mesh, "Lagrange", 2))
        u.vector()[:] = 1.0
        File("u.xdmf") << u

    def xtest_save_3d_vector(self):
        mesh = UnitCube(8, 8, 8)
        u = Function(VectorFunctionSpace(mesh, "Lagrange", 2))
        u.vector()[:] = 1.0
        File("u.xdmf") << u

    def xtest_save_3d_vector_series(self):
        mesh = UnitCube(8, 8, 8)
        u = Function(VectorFunctionSpace(mesh, "Lagrange", 2))
        file = File("u_3D.xdmf")

        u.vector()[:] = 1.0
        file << (u, 0.1)

        u.vector()[:] = 2.0
        file << (u, 0.2)

        #u.vector()[:] = 3.0
        #file << (u, 0.3)

    def xtest_save_2d_tensor(self):
        mesh = UnitSquare(16, 16)
        u = Function(TensorFunctionSpace(mesh, "Lagrange", 2))
        u.vector()[:] = 1.0
        File("u.xdmf") << u

    def xtest_save_3d_tensor(self):
        mesh = UnitCube(8, 8, 8)
        u = Function(TensorFunctionSpace(mesh, "Lagrange", 2))
        u.vector()[:] = 1.0
        File("u.xdmf") << u

if __name__ == "__main__":
    if has_hdf5():
        unittest.main()
