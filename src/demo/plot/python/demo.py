__author__ = "Anders Logg (logg@simula.no)"
__date__ = "2007-05-29 -- 2007-05-29"
__copyright__ = "Copyright (C) 2007 Anders Logg"
__license__  = "GNU GPL Version 2"

from dolfin import *

mesh = Mesh("cow.xml.gz")
plot(mesh)
