"""Unit tests for the HDF5 io library"""

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
# Modified by Chris Richardson 2013
#
# First added:  2012-09-14
# Last changed: 2013-05-24

import unittest
from dolfin import *

if has_hdf5():
    class HDF5_Vector(unittest.TestCase):
        """Test input/output of Vector to HDF5 files"""

        def test_save_vector(self):
            x = Vector(305)
            x[:] = 1.0
            vector_file = HDF5File("x.h5", "w")
            vector_file.write(x, "/my_vector")

        def test_save_and_read_vector(self):
            # Write to file
            x = Vector(305)
            x[:] = 1.2
            vector_file = HDF5File("vector.h5", "w")
            vector_file.write(x, "/my_vector")
            del vector_file

            # Read from file
            y = Vector()
            vector_file = HDF5File("vector.h5", "r")
            vector_file.read(y, "/my_vector")
            self.assertEqual(y.size(), x.size())
            self.assertEqual((x - y).norm("l1"), 0.0)

    class HDF5_MeshFunction(unittest.TestCase):

        def test_save_and_read_meshfunction_2D(self):
            # Write to file
            mesh = UnitSquareMesh(20,20)
            mf_file = HDF5File("meshfn-2d.h5", "w")

            # save meshfuns to compare when reading back
            meshfunctions = []
            for i in range(0,3):
                mf = MeshFunction('double', mesh, i)
                # NB choose a value to set which will be the same
                # on every process for each entity
                for cell in entities(mesh, i):
                    mf[cell] = cell.midpoint()[0]
                meshfunctions.append(mf)
                mf_file.write(mf, "/meshfunction/meshfun%d"%i)

            del mf_file

            # Read back from file
            mf_file = HDF5File("meshfn-2d.h5", "r")
            for i in range(0,3):
                mf2 = MeshFunction('double', mesh, i)
                mf_file.read(mf2, "/meshfunction/meshfun%d"%i)
                for cell in entities(mesh, i):
                    self.assertEqual(meshfunctions[i][cell], mf2[cell])

        def test_save_and_read_meshfunction_3D(self):
            # Write to file
            mesh = UnitCubeMesh(10, 10, 10)
            mf_file = HDF5File("meshfn-3d.h5", "w")

            # save meshfuns to compare when reading back
            meshfunctions = []
            for i in range(0,4):
                mf = MeshFunction('double', mesh, i)
                # NB choose a value to set which will be the same
                # on every process for each entity
                for cell in entities(mesh, i):
                    mf[cell] = cell.midpoint()[0]
                meshfunctions.append(mf)
                mf_file.write(mf, "/meshfunction/group/%d/meshfun"%i)

            del mf_file

            # Read back from file
            mf_file = HDF5File("meshfn-3d.h5", "r")
            for i in range(0,4):
                mf2 = MeshFunction('double', mesh, i)
                mf_file.read(mf2, "/meshfunction/group/%d/meshfun"%i)
                for cell in entities(mesh, i):
                    self.assertEqual(meshfunctions[i][cell], mf2[cell])

    class HDF5_MeshValueCollection(unittest.TestCase):

        def test_save_and_read_mesh_value_collection(self):
            mesh = UnitCubeMesh(5, 5, 5)

            # Writ to file
            hdf5_file = HDF5File("mesh_value_collection.h5", "w")
            for dim in range(mesh.topology().dim()):
                mvc = MeshValueCollection("size_t", mesh, dim)
                for i, cell in enumerate(entities(mesh, dim)):
                    mvc.set_value(cell.index(), i)
                hdf5_file.write(mvc, "/mesh_value_collection_%d"%dim)
            del hdf5_file

            # Read from file
            hdf5_file = HDF5File("mesh_value_collection.h5", "r")
            for dim in range(mesh.topology().dim()):
                mvc = MeshValueCollection("size_t", mesh, dim)
                hdf5_file.read(mvc, "/mesh_value_collection_%d"%dim)


    class HDF5_Mesh(unittest.TestCase):

        def test_save_and_read_mesh_2D(self):
            # Write to file
            mesh0 = UnitSquareMesh(20, 20)
            mesh_file = HDF5File("mesh.h5", "w")
            mesh_file.write(mesh0, "/my_mesh")
            del mesh_file

            # Read from file
            mesh1 = Mesh()
            mesh_file = HDF5File("mesh.h5", "r")
            mesh_file.read(mesh1, "/my_mesh")

            self.assertEqual(mesh0.size_global(0), mesh1.size_global(0))
            dim = mesh0.topology().dim()
            self.assertEqual(mesh0.size_global(dim), mesh1.size_global(dim))

        def test_save_and_read_mesh_3D(self):
            # Write to file
            mesh0 = UnitCubeMesh(10, 10, 10)
            mesh_file = HDF5File("mesh.h5", "w")
            mesh_file.write(mesh0, "/my_mesh")
            del mesh_file

            # Read from file
            mesh1 = Mesh()
            mesh_file = HDF5File("mesh.h5", "r")
            mesh_file.read(mesh1, "/my_mesh")

            self.assertEqual(mesh0.size_global(0), mesh1.size_global(0))
            dim = mesh0.topology().dim()
            self.assertEqual(mesh0.size_global(dim), mesh1.size_global(dim))


if __name__ == "__main__":
    unittest.main()
