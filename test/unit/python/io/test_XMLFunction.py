#!/usr/bin/env py.test
"""Unit tests for the XML input/output of Function"""

# Copyright (C) 2014 Matthias Liertzer
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

from dolfin import *
from dolfin_utils.test import cd_tempdir

def test_save_and_read_xml_function(cd_tempdir):
    mesh = UnitSquareMesh(10, 10)
    Q = FunctionSpace(mesh, "CG", 3)
    F0 = Function(Q)
    F1 = Function(Q)
    E = Expression("x[0]", degree=1)
    F0.interpolate(E)

    # Save to XML File
    xml_file = File("function.xml")
    xml_file << F0
    del xml_file

    # Read back from XML File
    xml_file = File("function.xml")
    xml_file >> F1
    result = F0.vector() - F1.vector()

    assert len(result.array().nonzero()[0]) == 0
