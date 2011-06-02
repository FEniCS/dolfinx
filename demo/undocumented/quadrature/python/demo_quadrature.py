#!/usr/bin/env python

# Copyright (C) 2007 Kristian B. Oelgaard
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
# First added:  2007-11-28
# Last changed: 2007-11-28

from dolfin import *
import sys

def main(argv):
    "Main function"

    if len(argv) != 2:
        print "Usage: './demo rule n' where rule is one of"
        print "gauss, radau, lobatto, and n is the number of points"
        return 2

    n = int(argv[1])

    if argv[0] == "gauss":
        q = GaussQuadrature(n)
        q.disp()

    elif argv[0] == "radau":
        q = RadauQuadrature(n)
        q.disp()

    elif argv[0] == "lobatto":
        q = LobattoQuadrature(n);
        q.disp();

    else:
        print "Unknown quadrature rule: %s" %argv[0]
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
