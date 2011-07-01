# Copyright (C) 2011 Marie E. Rognes
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
# First added:  2011-05-22
# Last changed: 2011-07-01

import sys
from dolfin_utils.commands import getstatusoutput

if len(sys.argv) != 2:
    usage = "Usage: python test.py absolute_path_to_dolfin_demo_dir"
    print usage
    sys.exit(2)

tests = ["verify_demo_code_snippets.py %s" % sys.argv[1]]

failed = []
for test in tests:
    command = "python %s" % test
    fail, output = getstatusoutput(command)

    if fail:
        failed.append(fail)
        print "*** %s failed" % test
        print output
    else:
        print "OK"

sys.exit(len(failed))
