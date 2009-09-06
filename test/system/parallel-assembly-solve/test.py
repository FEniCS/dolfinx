"Run solver.py in parallel"

__author__ = "Anders Logg (logg@simula.no)"
__date__ = "2009-08-17 -- 2009-08-17"
__copyright__ = "Copyright (C) 2009 Anders Logg"
__license__  = "GNU LGPL version 2.1"

import sys
from dolfin.utils import getoutput

# If it works on nine processes, it works for any number of processes (N. Jansson, 2009)
num_processes = 2 # Should really be 9 here, but the JIT compilation takes forever

# Run solver.py
output = getoutput("mpirun -n %d python solver.py" % num_processes)
if len(sys.argv) > 1 and sys.argv[1] == "--debug":
    print output

# Return exit status
if "ERROR" in output:
    sys.exit(1)
else:
    sys.exit(0)
