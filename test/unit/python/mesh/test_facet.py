"""Unit tests for the Facet class"""

# Copyright (C) 2017 Tormod Landet
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

from dolfin import UnitSquareMesh, facets


def test_normal():
    "Test that the normal() method is wrapped"
    mesh = UnitSquareMesh(4, 4)
    for facet in facets(mesh):
        n = facet.normal()
        nx, ny, nz = n.x(), n.y(), n.z()
        assert isinstance(nx, float)
        assert isinstance(ny, float)
        assert isinstance(nz, float)
