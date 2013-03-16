"""Unit test for HarmonicSmoothing and ALE"""

# Copyright (C) 2013 Jan Blechta
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
# First added:  2013-03-02
# Last changed: 2013-03-04

import unittest
from dolfin import UnitSquareMesh, BoundaryMesh, Expression, \
                   CellFunction, SubMesh, Constant, MPI

class HarmonicSmoothingTest(unittest.TestCase):

    def test_HarmonicSmoothing(self):

        print ""
        print "Testing HarmonicSmoothing::move(Mesh& mesh, " \
              "const BoundaryMesh& new_boundary)"

        # Create some mesh and its boundary
        mesh = UnitSquareMesh(10, 10)
        boundary = BoundaryMesh(mesh, 'exterior')

        # Move boundary
        disp = Expression(("0.3*x[0]*x[1]", "0.5*(1.0-x[1])"))
        boundary.move(disp)

        # Move mesh according to given boundary
        mesh.move(boundary)

        # Check that new boundary topology corresponds to given one
        boundary_new = BoundaryMesh(mesh, 'exterior')
        self.assertEqual(boundary.topology().hash(),
                         boundary_new.topology().hash())

        # Check that coordinates are almost equal
        err = sum(sum(abs(boundary.coordinates() \
                        - boundary_new.coordinates()))) / mesh.num_vertices()
        print "Current CG solver produced error in boundary coordinates", err
        self.assertAlmostEqual(err, 0.0, places=5)

        # Check mesh quality
        magic_number = 0.35
        self.assertTrue(mesh.radius_ratio_min()>magic_number)

if MPI.num_processes() == 1:

    class ALETest(unittest.TestCase):

        def test_ale(self):

            print ""
            print "Testing ALE::move(Mesh& mesh0, const Mesh& mesh1)"

            # Create some mesh
            mesh = UnitSquareMesh(4, 5)

            # Make some cell function
            # FIXME: Initialization by array indexing is probably
            #        not a good way for parallel test
            cellfunc = CellFunction('size_t', mesh)
            cellfunc.array()[0:4] = 0
            cellfunc.array()[4:]  = 1

            # Create submeshes - this does not work in parallel
            submesh0 = SubMesh(mesh, cellfunc, 0)
            submesh1 = SubMesh(mesh, cellfunc, 1)

            # Move submesh0
            disp = Constant(("0.1", "-0.1"))
            submesh0.move(disp)

            # Move and smooth submesh1 accordignly
            submesh1.move(submesh0)

            # Move mesh accordingly
            parent_vertex_indices_0 = \
                     submesh0.data().mesh_function('parent_vertex_indices')
            parent_vertex_indices_1 = \
                     submesh1.data().mesh_function('parent_vertex_indices')
            mesh.coordinates()[parent_vertex_indices_0.array()[:]] = \
                     submesh0.coordinates()[:]
            mesh.coordinates()[parent_vertex_indices_1.array()[:]] = \
                     submesh1.coordinates()[:]

            # If test passes here then it is probably working
            # Check for cell quality for sure
            magic_number = 0.28
            self.assertTrue(mesh.radius_ratio_min() > magic_number)


if __name__ == "__main__":
    print ""
    print "Testing HarmonicSmoothing and ALE operations"
    print "------------------------------------------------"
    unittest.main()
