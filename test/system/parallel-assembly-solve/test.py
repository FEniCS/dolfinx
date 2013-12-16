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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
#
# Modified by Johan Hake
# Modified by Johannes Ring 2011
# Modified by Garth N. Wells 2013
#
# First added:  2009-08-17
# Last changed: 2013-07-06

import sys
import subprocess
from dolfin import has_mpi, has_parmetis, has_scotch

if not (has_mpi()):
    print "DOLFIN has not been compiled with MPI. Test is not run."
    sys.exit(0)
elif not (has_parmetis() or has_scotch()):
    print "DOLFIN has not been compiled with ParMETIS or SCOTCH. Test is not run."
    sys.exit(0)

# Number of processes
num_processes = 3

# Run solver.py
output = subprocess.check_output(['mpirun', '-np', str(num_processes),
                                  sys.executable, 'solver.py'])
if len(sys.argv) > 1 and sys.argv[1] == "--debug":
    print output

# Return exit status
if "ERROR" in output:
    print output
    sys.exit(1)
else:
    print "OK"
    sys.exit(0)
