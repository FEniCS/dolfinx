"This demo illustrates how to plot a finite element."

__author__ = "Anders Logg (logg@simula.no)"
__date__ = "2010-12-08 -- 2010-12-08"
__copyright__ = "Copyright (C) 2010 Anders Logg"
__license__  = "GNU LGPL Version 2.1"

from dolfin import *

element = FiniteElement("Brezzi-Douglas-Marini", tetrahedron, 3)
plot(element)
