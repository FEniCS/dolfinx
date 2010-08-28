"""Run all unit tests."""

__author__ = "Anders Logg (logg@simula.no)"
__date__ = "2006-08-09 -- 2009-11-16"
__copyright__ = "Copyright (C) 2006-2007 Anders Logg"
__license__  = "GNU LGPL Version 2.1"

# Modified by Johannes Ring 2009

import sys, os, re
import platform
from dolfin_utils.commands import getoutput
from dolfin import has_mpi, has_parmetis

# Tests to run
tests = ["fem", "function", "mesh", "meshconvert", "la", "io", "python-extras", "quadrature"]

# Tests only available in Python
only_python = ["python-extras"]

# FIXME: Graph tests disabled for now since SCOTCH is now required

# Check if we should run only Python tests, use for quick testing
if len(sys.argv) == 2 and sys.argv[1] == "--only-python":
    only_python = tests

# Build prefix list
prefixes = [""]
if has_mpi() and has_parmetis():
    prefixes.append("mpirun -np 2 ")
else:
    print "DOLFIN has not been compiled with mpi and Parmetis. Unit tests will not be run in parallel."

# Run in serial, then in parallel
for prefix in prefixes:

    # Run tests
    failed = []
    for test in tests:
        print "Running unit tests for %s with prefix '%s'" % (test,  prefix)
        print "----------------------------------------------------------------------"

        cpptest_ext = ''
        if platform.system() == 'Windows':
            cpptest_ext = '.exe'
        print "C++:   ",
        if not test in only_python:
            output = getoutput("cd %s%scpp && %s .%s%s-test%s" % \
                               (test, os.path.sep, prefix,
                                os.path.sep, test, cpptest_ext))
            if "OK" in output:
                num_tests = int(re.search("OK \((\d+)\)", output).groups()[0])
                print "OK (%d tests)" % num_tests
            else:
                print "*** Failed"
                failed += [(test, "C++", output)]
        else:
            print "Skipping"

        print "Python:",
        output = getoutput("cd %s%spython && %s python .%stest.py" % \
                           (test, os.path.sep, prefix, os.path.sep))
        if "OK" in output:
            num_tests = int(re.search("Ran (\d+) test", output).groups()[0])
            print "OK (%d tests)" % num_tests
        else:
            print "*** Failed"
            failed += [(test, "Python", output)]

        print ""

    # Print output for failed tests
    for (test, interface, output) in failed:
        print "One or more unit tests failed for %s (%s):" % (test, interface)
        print output

# Return error code if tests failed
sys.exit(len(failed) != 0)
