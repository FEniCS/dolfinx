"""This program is used to generate the coefficients c00, c01 and c11
used in the demo."""

__author__ = "Anders Logg (logg@simula.no)"
__date__ = "2009-12-16"
__copyright__ = "Copyright (C) 2007-2009 Anders Logg"
__license__  = "GNU LGPL Version 2.1"

# Last changed: 2009-12-16

from dolfin import *

# Create mesh
mesh = UnitSquare(32, 32)

# Create mesh functions for c00, c01, c11
c00 = MeshFunction("double", mesh, 2)
c01 = MeshFunction("double", mesh, 2)
c11 = MeshFunction("double", mesh, 2)

# Iterate over mesh and set values
for cell in cells(mesh):
    if cell.midpoint().x() < 0.5:
        c00[cell] = 1.0
        c01[cell] = 0.3
        c11[cell] = 2.0
    else:
        c00[cell] = 3.0
        c01[cell] = 0.5
        c11[cell] = 4.0

# Store to file
mesh_file = File("mesh.xml.gz")
c00_file = File("c00.xml.gz")
c01_file = File("c01.xml.gz")
c11_file = File("c11.xml.gz")

mesh_file << mesh
c00_file << c00
c01_file << c01
c11_file << c11

# Plot mesh functions
plot(c00, title="C00")
plot(c01, title="C01")
plot(c11, title="C11")

interactive()
