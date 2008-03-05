#!/usr/bin/env python
#
# This simple demo generates different quadrature rules
#

__author__ = "Kristian B. Oelgaard (k.b.oelgaard@tudelft.nl)"
__date__ = "2007-11-28 -- 2007-11-28"
__copyright__ = "Copyright (C) 2007 Kristian B. Oelgaard"
__license__  = "GNU LGPL Version 2.1"

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
