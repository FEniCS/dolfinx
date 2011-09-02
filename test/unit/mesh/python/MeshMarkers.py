"""Unit tests for the class MeshMarkers. These unit test actually test
a bit more than that, since they also test marking of meshes and the
interaction between Mesh - MeshDomains - MeshMarkers"""

# Copyright (C) 2011 Anders Logg
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
# First added:  2011-09-01
# Last changed: 2011-09-02

import unittest
from dolfin import *

class XMLMesMarkers(unittest.TestCase):

    def test_subdomain_marking(self):
        "Test setting markers from subdomains"

        # Define subdomains for 5 of the 6 faces of the unit cube
        class F0(SubDomain):
            def inside(self, x, inside):
                return near(x[0], 0.0)
        class F1(SubDomain):
            def inside(self, x, inside):
                return near(x[0], 1.0)
        class F2(SubDomain):
            def inside(self, x, inside):
                return near(x[1], 0.0)
        class F3(SubDomain):
            def inside(self, x, inside):
                return near(x[1], 1.0)
        class F4(SubDomain):
            def inside(self, x, inside):
                return near(x[2], 0.0)
        f0 = F0()
        f1 = F1()
        f2 = F2()
        f3 = F3()
        f4 = F4()

        # Apply markers to unit cube
        mesh = UnitCube(3, 3, 3)
        f0.mark_facets(mesh, 0)
        f1.mark_facets(mesh, 1)
        f2.mark_facets(mesh, 2)
        f3.mark_facets(mesh, 3)
        f4.mark_facets(mesh, 4)

        # Write to file (used as input for another unit test)
        f = File("MeshMarkers_test_subdomain_marking.xml")
        f << mesh

        # FIXME: Add test here
        self.assertEqual(0, 0)

if __name__ == "__main__":
    unittest.main()
