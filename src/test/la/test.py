"""Run all linear algerbra tests"""

__author__ = "Garth N. Wells (g.n.wells@tudelft.nl)"
__date__ = "2006-08-11"
__copyright__ = "Copyright (C) 2006 Garth N. Wells"
__license__  = "GNU GPL Version 2"

from createUBlas import *
from algebraUBlas import *
from solverUBlas import *

if __name__ == '__main__':
    unittest.main()
