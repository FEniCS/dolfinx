"""Unit tests for the io library"""

# Copyright (C) 2007 Anders Logg
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
# First added:  2009-01-02
# Last changed: 2009-01-02

import unittest
from dolfin import *

# FIXME: Not a proper unit test. When LocalMeshData has a public interface
# FIXME: we can expand on these
class LocalMeshDataXML_IO(unittest.TestCase):

    def testRead(self):
        file = File("../../../../data/meshes/snake.xml.gz")
        localdata = cpp.LocalMeshData()
        file >> localdata

if __name__ == "__main__":
    unittest.main()
