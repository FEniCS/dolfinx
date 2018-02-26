"This demo illustrates how to plot a finite element."

# Copyright (C) 2010 Anders Logg
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from dolfin import *

element = FiniteElement("Brezzi-Douglas-Marini", tetrahedron, 3)
plot(element)
