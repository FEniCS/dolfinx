"Unit tests for the mesh library"

# Copyright (C) 2006 Anders Logg
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
# First added:  2013-02-06
# Last changed: 2013-02-06

import unittest
from dolfin import *

# FIXME: The following test breaks in parallel
if MPI.num_processes() == 1:
    class SubMeshTester(unittest.TestCase):

        def test_creation(self):
            """Create SubMesh."""
            for MeshClass, args in [(UnitIntervalMesh, (10,)),
                                    (UnitSquareMesh, (10, 10)),
                                    (UnitCubeMesh, (10,10,10))]:
                
                mesh = MeshClass(*args)
                dim_t = mesh.topology().dim()
                mesh.domains().init(dim_t)
                markers = mesh.domains().markers(dim_t)
                domains = CellFunction("size_t", mesh, 0)
                for cell in cells(mesh):
                    # Mark half the cells
                    if cell.index()>mesh.num_cells()/2:
                        break
                    domains[cell] = 1
                    markers.set_value(cell.index(), 0, 1)

                # Create mesh from stored MeshValueCollection and external CellFunction
                smesh0 = SubMesh(mesh, 1)
                smesh1 = SubMesh(mesh, domains, 1)

                self.assertEqual(smesh0.num_cells(), smesh1.num_cells())
                self.assertEqual(smesh0.num_vertices(), smesh1.num_vertices())

                # Check that we create the same sub mesh with the same MeshValueCollection
                for cell0, cell1 in zip(cells(smesh0), cells(smesh1)):
                    self.assertEqual(cell0.index(), cell1.index())
                    self.assertEqual(smesh0.domains().markers(dim_t).get_value(cell0.index(), 0),
                                     smesh1.domains().markers(dim_t).get_value(cell1.index(), 0))

                self.assertRaises(RuntimeError, SubMesh, (mesh, 2))
                mesh = MeshClass(*args)
                self.assertRaises(RuntimeError, SubMesh, (mesh, 1))

        def test_facet_domain_propagation(self):
            # Boxes contains two subdomains with marked faces between them.
            # These faces are marked with 5, 10, 15. 
            mesh = Mesh("../boxes.xml.gz")
            inner = SubMesh(mesh, 1)
            outer = SubMesh(mesh, 2)

            parent_facets = mesh.domains().facet_domains()
            inner_facets = inner.domains().facet_domains()
            outer_facets = outer.domains().facet_domains()

            # Check we have the same number of value-marked facets
            for value in [5, 10, 15]:
                self.assertEqual((inner_facets.array()==value).sum(),
                                 (outer_facets.array()==value).sum())
                self.assertEqual((parent_facets.array()==value).sum(),
                                 (outer_facets.array()==value).sum())
            

if __name__ == "__main__":
    unittest.main()
