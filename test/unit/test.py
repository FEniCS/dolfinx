"""Run all unit tests."""

__author__ = "Anders Logg (logg@simula.no)"
__date__ = "2006-08-09 -- 2011-03-22"
__copyright__ = "Copyright (C) 2006-2007 Anders Logg"
__license__  = "GNU LGPL Version 2.1"

# Modified by Johannes Ring 2009, 2011

import sys, os, re
import platform
from dolfin_utils.commands import getstatusoutput
from dolfin import has_mpi, has_parmetis

# Tests to run
tests = {
    "fem": ["test", "Assembly", "DirichletBC"],
    "function": ["test", "function"],
    "mesh": ["test", "MeshFunction", "Edge", "Face"],
    "meshconvert": ["test"],
    "la": ["test", "Vector", "Matrix"],
    "io": ["test"],
    "parameter": ["test"],
    "python-extras": ["test"],
    "quadrature": ["test"],
    "adaptivity": ["errorcontrol"]
    }

# FIXME: Graph tests disabled for now since SCOTCH is now required

# Run both C++ and Python tests as default
only_python = False

# Check if we should run only Python tests, use for quick testing
if len(sys.argv) == 2 and sys.argv[1] == "--only-python":
    only_python = True

# Build prefix list
prefixes = [""]
if has_mpi() and has_parmetis():
    prefixes.append("mpirun -np 3 ")
else:
    print "DOLFIN has not been compiled with MPI and/or ParMETIS. Unit tests will not be run in parallel."

# Run in serial, then in parallel
for prefix in prefixes:

    # Run tests
    failed = []
    for test, subtests in tests.items():
        for subtest in subtests:
            print "Running unit tests for %s (%s) with prefix '%s'" % (test,  subtest, prefix)
            print "----------------------------------------------------------------------"

            cpptest_executable = subtest + "_" + test
            if platform.system() == 'Windows':
                cpptest_executable += '.exe'
            print "C++:   ",
            if not only_python and os.path.isfile(os.path.join(test, "cpp", cpptest_executable)):
                status, output = getstatusoutput("cd %s%scpp && %s .%s%s" % \
                                   (test, os.path.sep, prefix, os.path.sep, cpptest_executable))
                if status == 0 and "OK" in output:
                    num_tests = int(re.search("OK \((\d+)\)", output).groups()[0])
                    print "OK (%d tests)" % num_tests
                else:
                    print "*** Failed"
                    failed += [(test, subtest, "C++", output)]
            else:
                print "Skipping"

            print "Python:",
            if os.path.isfile(os.path.join(test, "python", subtest + ".py")):
                status, output = getstatusoutput("cd %s%spython && %s python .%s%s.py" % \
                                   (test, os.path.sep, prefix, os.path.sep, subtest))
                if status == 0 and "OK" in output:
                    num_tests = int(re.search("Ran (\d+) test", output).groups()[0])
                    print "OK (%d tests)" % num_tests
                else:
                    print "*** Failed"
                    failed += [(test, subtest, "Python", output)]
            else:
                print "Skipping"

            print ""

    # Print output for failed tests
    for (test, subtest, interface, output) in failed:
        print "One or more unit tests failed for %s (%s, %s):" % (test, subtest, interface)
        print output

# Return error code if tests failed
sys.exit(len(failed) != 0)
