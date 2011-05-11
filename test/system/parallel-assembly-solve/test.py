"Run solver.py in parallel"

# Copyright (C) 2009 Anders Logg
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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
#
# Modified by Johan Hake
# Modified by Johannes Ring 2011
#
# First added:  2009-08-17
# Last changed: 2011-03-12

import sys
from dolfin_utils.commands import getstatusoutput
from dolfin import has_mpi, has_parmetis

if not (has_mpi() and has_parmetis()):
    print "DOLFIN has not been compiled with mpi and Parmetis. Test is not run."
    sys.exit(0)

# Number of processes
num_processes = 3

# Run solver.py
failure, output = getstatusoutput("mpirun -n %d python solver.py" % num_processes)
if len(sys.argv) > 1 and sys.argv[1] == "--debug":
    print output

# Return exit status
if "ERROR" in output:
    print output
    sys.exit(1)
else:
    print "OK"
    sys.exit(0)
