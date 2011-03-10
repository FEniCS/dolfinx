"Run all system tests"

__author__ = "Anders Logg (logg@simula.no)"
__date__ = "2009-08-17 -- 2011-03-09"
__copyright__ = "Copyright (C) 2009 Anders Logg"
__license__  = "GNU LGPL version 2.1"

# Modified by Johannes Ring 2011

import os, sys
from dolfin_utils.commands import getstatusoutput

pwd = os.path.dirname(os.path.abspath(__file__))

# Tests to run
tests = ["parallel-assembly-solve"]

failed = []

# Command to run
command = "python test.py" + " " + " ".join(sys.argv[1:])

# Run tests
for test in tests:
    print "Running system test: %s" % test
    print "----------------------------------------------------------------------"
    os.chdir(os.path.join(pwd, test))
    fail, output = getstatusoutput(command)
    if fail:
        failed.append(fail)
        print "*** Failed"
        print output
    else:
        print "OK"

sys.exit(len(failed))
