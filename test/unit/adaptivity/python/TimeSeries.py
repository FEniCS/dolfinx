"""Unit tests for TimeSeries"""

# Copyright (C) 2011 Marie E. Rognes
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
#
# First added:  2011-06-16
# Last changed: 2011-06-16

import unittest
#from unittest import skipIf # Awaiting Python 2.7
from dolfin import *

#@skipIf("Skipping TimeSeries test in parallel", MPI.num_processes() > 1)
class TimeSeriesTest(unittest.TestCase):

    def test_retrieve_compressed(self):
        self._test_retrieve(True, False)

    def test_retrieve_compressed_all_connectivities(self):
        self._test_retrieve(True, True)

    def test_retrieve_all_connectivities(self):
        self._test_retrieve(False, True)

    def _test_retrieve(self, compressed=False, all_connectivities=False):

        if MPI.num_processes() > 1:
            return

        times = [t/10.0 for t in range(1, 11)]

        mesh_size = (2, 2, 2)
        mesh = UnitCube(*mesh_size)
        mesh.init()
        V = FunctionSpace(mesh, "CG", 2)

        u = Function(V)
        series = TimeSeries("u", compressed, all_connectivities)
        for t in times:
            u.vector()[:] = t
            series.store(u.vector(), t)
            series.store(mesh, t)

        series = TimeSeries("u", compressed)
        t0 = series.vector_times()[0]
        T = series.mesh_times()[-1]

        self.assertAlmostEqual(t0, times[0])
        self.assertAlmostEqual(T, times[-1])

        # Test retreiving of mesh
        mesh_retreived = Mesh()
        series.retrieve(mesh_retreived, 0.1)

        mesh_test = mesh if all_connectivities else UnitCube(*mesh_size)

        for entity in range(4):
            self.assertEqual(mesh_retreived.topology().size(entity),
                             mesh_test.topology().size(entity))

        for i in range(4):
            for j in range(4):
                self.assertEqual(mesh_retreived.topology()(i, j).size(),
                                 mesh_test.topology()(i, j).size())

if __name__ == "__main__":
    print ""
    print "Testing TimeSeries operations"
    print "------------------------------------------------"
    unittest.main()
