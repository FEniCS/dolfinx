"""This program is used to generate the coefficients c00, c01 and c11
used in the demo."""

# Copyright (C) 2007-2009 Anders Logg
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from dolfin import *

# Create mesh
mesh = UnitSquareMesh(32, 32)

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
mesh_file = File("../unitsquare_32_32.xml.gz")
c00_file = File("../unitsquare_32_32_c00.xml.gz")
c01_file = File("../unitsquare_32_32_c01.xml.gz")
c11_file = File("../unitsquare_32_32_c11.xml.gz")

mesh_file << mesh
c00_file << c00
c01_file << c01
c11_file << c11

# Plot mesh functions
plot(c00, title="C00", interactive=True)
plot(c01, title="C01", interactive=True)
plot(c11, title="C11", interactive=True)
