#!/usr/bin/env py.test

"""Unit tests for the XML io library for Tables"""

# Copyright (C) 2016 Simon Funke and Marie E. Rognes
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
from dolfin_utils.test import cd_tempdir

def test_write_and_read_table(cd_tempdir):
    # Do something that takes time
    x = PETScVector(mpi_comm_world(), 197)

    if MPI.rank(mpi_comm_world()) == 0:
        # Create table for timings
        t = timings(TimingClear_keep, [TimingType_wall, TimingType_system])
        t_str = t.str(True)

        # Write table to file
        file = File(mpi_comm_self(), "my_table.xml")
        file << t
        del t
        del file
    
        # Read table from file
        file = File(mpi_comm_self(), "my_table.xml")
        t = Table("My Table")
        file >> t

        t_new_str = t.str(True)

        assert t_new_str == t_str
