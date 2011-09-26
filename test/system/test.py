"Run all system tests"

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
# Modified by Johannes Ring 2011
#
# First added:  2009-08-17
# Last changed: 2011-03-09

import os, sys
from dolfin_utils.commands import getstatusoutput

pwd = os.path.dirname(os.path.abspath(__file__))

# Tests to run
tests = ["parallel-assembly-solve", "ufl-jit-assemble-chain"]

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
