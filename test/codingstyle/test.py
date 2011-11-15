"Run all tests"

# Copyright (C) 2011 Anders Logg
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
# First added:  2011-11-14
# Last changed: 2011-11-14

import os, re

cpp_tests = []
python_tests = []

def test_codingstyle(topdir, language, subdir, suffixes, tests):
    "Main function for performing tests"

    print "Running tests for %s coding style, watch out..." % language
    print "----------------------------------------------------"

    # Iterate over all files
    num_failed = 0
    for subdir, dirs, files in os.walk(os.path.join(topdir, subdir)):
        for filename in files:

            # Only consider files with given suffix
            if not len([1 for suffix in suffixes if filename.endswith(suffix)]) > 0:
                continue

            # Read file
            f = open(os.path.join(subdir, filename), "r")
            code = f.read()
            f.close()

            # Perform all tests
            for test in tests:
                ok = test(code, filename)
                if not ok:
                    num_failed += 1

    # Print summary
    print
    print "Ran %d test(s)" % len(tests)
    if num_failed == 0:
        print "OK"
    else:
        print "*** %d tests failed" % num_failed
    print

def test_dolfin_error(code, filename):
    "Test for use of dolfin_error vs error"

    # Skip exceptions
    exceptions = ["log.h", "log.cpp", "Logger.h", "Logger.cpp",
                  "pugixml.cpp", "Poisson1D.h", "Poisson2D.h", "Poisson3D.h",
                  "meshconvert.py"]
    if filename in exceptions:
        return True

    # Check for error(...)
    if re.search(r"\berror\(", code) is None:
        return True

    # Write an error message
    print "*** error() used in %s when dolfin_error() should be used" % filename

    return False

# List of C++ tests
cpp_tests = [test_dolfin_error]

# List of Python tests
python_tests = [test_dolfin_error]

if __name__ == "__main__":

    # Set up paths
    pwd = os.path.dirname(os.path.abspath(__file__))
    topdir = os.path.join(pwd, "..", "..")

    # Check C++ files
    test_codingstyle(topdir, "C++", "dolfin", [".cpp", ".h"], cpp_tests)

    # Check Python files
    test_codingstyle(topdir, "Python", "site-packages", [".py"], python_tests)

#sys.exit(len(failed))
