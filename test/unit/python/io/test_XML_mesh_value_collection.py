"Unit tests for XML input/output of MeshValueCollection"

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

import pytest
from dolfin import *
import os
from dolfin_utils.test import fixture, skip_in_parallel, cd_tempdir, filedir

@skip_in_parallel
def test_insertion_extraction_io(cd_tempdir):
    "Test input/output via << and >>."
    filename = "xml_mesh_value_collection_test_io.xml"

    # Create mesh
    mesh = UnitCubeMesh(5, 5, 5)

    # Create mesh value collection and add some data
    mesh = UnitCubeMesh(5, 5, 5)
    output_values = MeshValueCollection("size_t", mesh, 2)
    output_values.set_value(1,  1, 1);
    output_values.set_value(2,  1, 3);
    output_values.set_value(5,  1, 8);
    output_values.set_value(13, 1, 21);
    output_values.set_value(7,  2, 13);
    output_values.set_value(4,  2, 2);

    name = "test_mesh_value_collection"
    output_values.rename(name, "a MeshValueCollection")

    # Write to file
    output_file = File(filename)
    output_file << output_values

    # Read from file
    input_file = File(filename)
    input_values = MeshValueCollection("size_t", mesh)
    input_file >> input_values

    # Get some data and check that it matches
    assert input_values.size() == output_values.size()
    assert input_values.dim() == output_values.dim()
    assert input_values.name() == name

def test_constructor_input(filedir):
    "Test input via constructor."
    filename = os.path.join(filedir, "xml_value_collection_ref.xml")

    # Create mesh
    mesh = UnitCubeMesh(5, 5, 5)

    # Read from file
    input_values = MeshValueCollection("size_t", mesh, filename)

    # Check that size is correct
    assert MPI.sum(mesh.mpi_comm(), input_values.size()) == 6
