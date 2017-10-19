"Unit tests for XML input/output of Mesh (class XMLMesh)"

# Copyright (C) 2011 Garth N. Wells
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
# Modified by Anders Logg 2011

import pytest
from dolfin import *
import os
from dolfin_utils.test import skip_in_parallel, fixture, cd_tempdir


@skip_in_parallel
def test_save_plain_mesh2D(cd_tempdir):
    mesh = UnitSquareMesh(8, 8)
    f = File("unit_square.xml")
    f << mesh

@skip_in_parallel
def test_save_plain_mesh3D(cd_tempdir):
    mesh = UnitCubeMesh(8, 8, 8)
    f = File("unit_cube.xml")
    f << mesh

@skip_in_parallel
def test_mesh_domains_io(cd_tempdir):
    "Test input/output for mesh domains"

    # Define subdomains for the 6 faces of the unit cube
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
    class F5(SubDomain):
        def inside(self, x, inside):
            return near(x[2], 1.0)

    # Define subdomains for left and right of x = 0.5
    class S0(SubDomain):
        def inside(self, x, inside):
            return x[0] < 0.5 + DOLFIN_EPS

    class S1(SubDomain):
        def inside(self, x, inside):
            return x[0] > 0.5 - DOLFIN_EPS

    # Create a mesh
    output_mesh = UnitCubeMesh(3, 3, 3)

    # Apply facet markers
    f0 = F0()
    f0.mark_facets(output_mesh, 0)
    f1 = F1()
    f1.mark_facets(output_mesh, 1)
    f2 = F2()
    f2.mark_facets(output_mesh, 2)
    f3 = F3()
    f3.mark_facets(output_mesh, 3)
    f4 = F4()
    f4.mark_facets(output_mesh, 4)
    f5 = F5()
    f5.mark_facets(output_mesh, 5)

    # Apply cell markers
    s0 = S0()
    s0.mark_cells(output_mesh, 0)
    s1 = S1()
    s1.mark_cells(output_mesh, 1)

    # Write to file
    output_file = File("XMLMesh_test_mesh_domains_io.xml")
    output_file << output_mesh

    # Read from file
    input_file = File("XMLMesh_test_mesh_domains_io.xml")
    input_mesh = Mesh()
    input_file >> input_mesh

    # Get some data and check that it matches
    assert len(input_mesh.domains().markers(0)) == \
            len(output_mesh.domains().markers(0))
    assert len(input_mesh.domains().markers(1)) == \
            len(output_mesh.domains().markers(1))
    assert len(input_mesh.domains().markers(2)) == \
            len(output_mesh.domains().markers(2))
    assert len(input_mesh.domains().markers(3)) == \
            len(output_mesh.domains().markers(3))
