# Copyright (C) 2026 Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""P1 mass matrix and load vector forms, compiled with FFCx.

These are the UFL/FFCx equivalents of the hand-written kernels
`kernel_a` and `kernel_L` in main.cpp: the P1 mass matrix and the RHS
load vector for f = 1 on a triangle. Compiling them gives a kernel
generated from UFL, exercised via the same low-level lambda-kernel
assembly path used for the hand-written kernel.

The `ffcx_kernel_name` metadata fixes the name of the generated kernel
function, so that main.cpp can call it directly. Without it FFCx derives
a name from the Python variable holding the form.
"""

from basix.ufl import element
from ufl import FunctionSpace, Mesh, TestFunction, TrialFunction, dx, inner

coord_element = element("Lagrange", "triangle", 1, shape=(2,))
mesh = Mesh(coord_element)

e = element("Lagrange", "triangle", 1)
V = FunctionSpace(mesh, e)

u = TrialFunction(V)
v = TestFunction(V)

a = inner(u, v) * dx(metadata={"ffcx_kernel_name": "tabulate_tensor_mass"})
L = inner(1.0, v) * dx(metadata={"ffcx_kernel_name": "tabulate_tensor_load"})
