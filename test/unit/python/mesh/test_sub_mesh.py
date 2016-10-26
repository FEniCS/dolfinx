#!/usr/bin/env py.test

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

import os
import pytest
from dolfin import *
import six
from dolfin_utils.test import skip_in_parallel, datadir


@pytest.fixture(scope='module', params=range(3))
def MeshFunc(request):
    test_mesh = [(UnitIntervalMesh, (10,)),
                 (UnitSquareMesh, (10, 10)),
                 (UnitCubeMesh, (10, 10, 10))]
    return test_mesh[request.param]


@skip_in_parallel
def test_creation(MeshFunc):
    """Create SubMesh."""
    args = MeshFunc[1]
    MeshFunc = MeshFunc[0]
    mesh = MeshFunc(*args)
    dim_t = mesh.topology().dim()
    mesh.domains().init(dim_t)
    domains = CellFunction("size_t", mesh, 0)
    for cell in cells(mesh):
        # Mark half the cells
        if cell.index() > mesh.num_cells()/2:
            break
        domains[cell] = 1
        mesh.domains().set_marker((cell.index(), 1), dim_t)

    # Create mesh from stored MeshValueCollection and
    # external CellFunction
    smesh0 = SubMesh(mesh, 1)
    smesh1 = SubMesh(mesh, domains, 1)
    assert smesh0.num_cells() == smesh1.num_cells()
    assert smesh0.num_vertices() == smesh1.num_vertices()
    # Check that we create the same sub mesh with the same
    # MeshValueCollection
    for cell0, cell1 in zip(cells(smesh0), cells(smesh1)):
        assert cell0.index() == cell1.index()
        assert smesh0.domains().get_marker(cell0.index(), dim_t) == \
            smesh1.domains().get_marker(cell1.index(), dim_t)

    # This test passed for unittest because it called SubMesh((mesh, 2))
    # and not SubMesh(mesh, 2)
    # with pytest.raises(RuntimeError):
    #    SubMesh(mesh, 2)

    mesh = MeshFunc(*args)
    with pytest.raises(RuntimeError):
        SubMesh(mesh, 1)


@skip_in_parallel
def test_facet_domain_propagation(datadir):

    # Boxes contains two subdomains with marked faces between
    # them.  These faces are marked with 5, 10, 15.
    mesh = Mesh(os.path.join(datadir, "boxes.xml"))
    inner = SubMesh(mesh, 1)
    outer = SubMesh(mesh, 2)

    # Test dict interface
    D = mesh.topology().dim() - 1
    parent_facets = mesh.domains().markers(D)
    inner_facets = inner.domains().markers(D)
    outer_facets = outer.domains().markers(D)

    for value in [5, 10, 15]:
        sum_parent = 0
        sum_inner = 0
        sum_outer = 0
        for key, val in six.iteritems(parent_facets):
            if val == value:
                sum_parent += val
        for key, val in six.iteritems(inner_facets):
            if val == value:
                sum_inner += val
        for key, val in six.iteritems(outer_facets):
            if val == value:
                sum_outer += val

        assert sum_outer == sum_inner
        assert sum_outer == sum_parent

    # Test Meshfunction interface
    parent_facets = MeshFunction("size_t", mesh, D, mesh.domains())
    inner_facets = MeshFunction("size_t", inner, D, inner.domains())
    outer_facets = MeshFunction("size_t", outer, D, outer.domains())

    # Check we have the same number of value-marked facets
    for value in [5, 10, 15]:
        assert ((inner_facets.array() == value).sum() ==
                (outer_facets.array() == value).sum())
        assert ((parent_facets.array() == value).sum() ==
                (outer_facets.array() == value).sum())
