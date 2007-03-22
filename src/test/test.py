"""Run all unit tests."""

__author__ = "Anders Logg (logg@simula.no)"
__date__ = "2006-08-09 -- 2007-03-22"
__copyright__ = "Copyright (C) 2006-2007 Anders Logg"
__license__  = "GNU GPL Version 2"

import unittest

from mesh.test import *
from la.test import *
from fem.test import *
from graph.test import *

unittest.main()
