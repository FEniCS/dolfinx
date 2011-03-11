"Run solver.py in parallel"

__author__ = "Anders Logg (logg@simula.no)"
__date__ = "2009-08-17 -- 2011-03-09"
__copyright__ = "Copyright (C) 2009 Anders Logg"
__license__  = "GNU LGPL version 2.1"

# Modified by Johan Hake
# Modified by Johannes Ring 2011

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
if failure:
    print output
    sys.exit(1)
else:
    print "OK"
    sys.exit(0)
