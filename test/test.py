"""Run all tests, including unit tests and regression tests"""

__author__ = "Anders Logg (logg@simula.no)"
__date__ = "2007-06-09 -- 2008-11-21"
__copyright__ = "Copyright (C) 2007-2008 Anders Logg"
__license__  = "GNU LGPL version 2.1"

import re, sys, os

pwd = os.path.dirname(os.path.abspath(__file__))

# Tests to run
#tests = ["unit", "system", "memory"]
tests = ["unit", "system"]#, "memory"]

failed = []

# Command to run
command = "python test.py" + " " + " ".join(sys.argv[1:])

# Run tests
for test in tests:
    print "Running tests: %s" % test
    print "----------------------------------------------------------------------"
    os.chdir(os.path.join(pwd, test))
    fail = os.system(command)
    if fail:
        failed.append(fail)
    print ""

sys.exit(len(failed))
