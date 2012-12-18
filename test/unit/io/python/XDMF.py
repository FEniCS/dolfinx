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
# Last changed: 2012-10-02

import unittest
from dolfin import *

if has_hdf5():
    class XDMF_Mesh_Output(unittest.TestCase):
        """Test output of Meshes to XDMF files"""

        def test_save_1d_mesh(self):
            mesh = UnitIntervalMesh(32)
            File("output/mesh.xdmf") << mesh

        def test_save_2d_mesh(self):
            mesh = UnitSquareMesh(32, 32)
            File("output/mesh_2D.xdmf") << mesh

        def test_save_3d_mesh(self):
            mesh = UnitCubeMesh(8, 8, 8)
            File("output/mesh_3D.xdmf") << mesh

    class XDMF_Vertex_Function_Output(unittest.TestCase):
        """Test output of vertex-based Functions to XDMF files"""

        def test_save_1d_scalar(self):
            mesh = UnitIntervalMesh(32)
            u = Function(FunctionSpace(mesh, "Lagrange", 2))
            u.vector()[:] = 1.0
            File("output/u.xdmf") << u

        def test_save_2d_scalar(self):
            mesh = UnitSquareMesh(16, 16)
            u = Function(FunctionSpace(mesh, "Lagrange", 2))
            u.vector()[:] = 1.0
            File("output/u.xdmf") << u

        def test_save_3d_scalar(self):
            mesh = UnitCubeMesh(8, 8, 8)
            u = Function(FunctionSpace(mesh, "Lagrange", 2))
            u.vector()[:] = 1.0
            File("output/u.xdmf") << u

        def test_save_2d_vector(self):
            mesh = UnitSquareMesh(16, 16)
            u = Function(VectorFunctionSpace(mesh, "Lagrange", 2))
            c = Constant((1.0, 2.0))
            u.interpolate(c)
            File("output/u_2dv.xdmf") << u

        def test_save_3d_vector(self):
            mesh = UnitCubeMesh(1, 1, 1)
            u = Function(VectorFunctionSpace(mesh, "Lagrange", 1))
            c = Constant((1.0, 2.0, 3.0))
            u.interpolate(c)
            File("output/u_3Dv.xdmf") << u

        def test_save_3d_vector_series(self):
            mesh = UnitCubeMesh(8, 8, 8)
            u = Function(VectorFunctionSpace(mesh, "Lagrange", 2))
            file = File("output/u_3D.xdmf")

            u.vector()[:] = 1.0
            file << (u, 0.1)

            u.vector()[:] = 2.0
            file << (u, 0.2)

            u.vector()[:] = 3.0
            file << (u, 0.3)

        def test_save_2d_tensor(self):
            mesh = UnitSquareMesh(16, 16)
            u = Function(TensorFunctionSpace(mesh, "Lagrange", 2))
            u.vector()[:] = 1.0
            File("output/tensor.xdmf") << u

        def test_save_3d_tensor(self):
            mesh = UnitCubeMesh(8, 8, 8)
            u = Function(TensorFunctionSpace(mesh, "Lagrange", 2))
            u.vector()[:] = 1.0
            File("output/u.xdmf") << u

    class XDMF_MeshFunction_Output(unittest.TestCase):
        """Test output of Meshes to XDMF files"""

        # FIXME: 1D seems to be broken
        def xtest_save_1d_mesh(self):
            mesh = UnitIntervalMesh(32)
            mf = CellFunction("size_t", mesh)
            for cell in cells(mesh):
                mf[cell] = cell.index()
            File("output/mf_1D.xdmf") << mf

        def test_save_2D_cell_function(self):
            mesh = UnitSquareMesh(32, 32)
            mf = CellFunction("size_t", mesh)
            for cell in cells(mesh):
                mf[cell] = cell.index()
            File("output/mf_2D.xdmf") << mf

        def test_save_3D_cell_function(self):
            mesh = UnitCubeMesh(8, 8, 8)
            mf = CellFunction("size_t", mesh)
            for cell in cells(mesh):
                mf[cell] = cell.index()
            File("output/mf_3D.xdmf") << mf

        def test_save_2D_facet_function(self):
            mesh = UnitSquareMesh(32, 32)
            mf = FacetFunction("size_t", mesh)
            for facet in facets(mesh):
                mf[facet] = facet.index()
            File("output/mf_facet_2D.xdmf") << mf

        def test_save_3D_facet_function(self):
            mesh = UnitCubeMesh(8, 8, 8)
            mf = FacetFunction("size_t", mesh)
            for facet in facets(mesh):
                mf[facet] = facet.index()
            File("output/mf_facet_3D.xdmf") << mf

if __name__ == "__main__":
    unittest.main()
