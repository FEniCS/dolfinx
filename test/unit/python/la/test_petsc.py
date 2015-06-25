#!/usr/bin/env py.test

"""Unit tests for parts of the PETSc interface not tested via the
GenericFoo interface"""

# Copyright (C) 2015 Garth N. Wells
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

from __future__ import print_function
import pytest
from dolfin import *
from dolfin_utils.test import skip_if_not_PETSc, skip_in_parallel

@skip_if_not_PETSc
def test_options_prefix():
    "Test set/get prefix option for PETSc objects"

    # Create empty vector
    x = PETScVector()

    # Prefix
    prefix = "test_vector_"

    # Set prefix
    x.set_options_prefix(prefix)

    # Get prefix (should be empty since vector has been initialised)
    assert not x.get_options_prefix()

    # Initialise vector
    x.init(mpi_comm_world(), 100)

    # Check prefix
    assert x.get_options_prefix() == prefix

    # Try changing prefix post-intialisation (should throw error)
    with pytest.raises(RuntimeError):
        x.set_options_prefix("test")
