"Run solver.py in parallel"

__author__ = "Anders Logg (logg@simula.no)"
__date__ = "2009-08-17 -- 2009-09-21"
__copyright__ = "Copyright (C) 2009 Anders Logg"
__license__  = "GNU LGPL version 2.1"

# Modified by Johan Hake

import sys
from dolfin_utils.commands import getoutput
from dolfin import has_mpi, has_parmetis

if not (has_mpi() and has_parmetis()):
    print "DOLFIN has not been compiled with mpi and Parmetis. Test is not run."
    sys.exit(0)

# If it works on nine processes, it works for any number of processes (N. Jansson, 2009)
num_processes = 9 # Should really be 9 here, but it breaks

# Run solver.py
output = getoutput("mpirun -n %d python solver.py" % num_processes)
if len(sys.argv) > 1 and sys.argv[1] == "--debug":
    print output

# Return exit status
if "ERROR" in output:
    sys.exit(1)
else:
    print "OK"
    sys.exit(0)
