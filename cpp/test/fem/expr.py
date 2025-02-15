# Copyright (C) 2025 Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import ufl
from basix import CellType
from basix.ufl import element

cell = CellType.hexahedron
mesh1 = ufl.Mesh(element("Lagrange", CellType.hexahedron, degree=1, shape=(3,)))
mesh2 = ufl.Mesh(element("Lagrange", CellType.hexahedron, degree=2, shape=(3,)))

e = element("Lagrange", CellType.hexahedron, degree=1)
V1, V2 = ufl.FunctionSpace(mesh1, e), ufl.FunctionSpace(mesh2, e)

u1, u2 = ufl.Coefficient(V1), ufl.Coefficient(V2)
pts = [[0.25, 0.25, 0.25]]
Q6_P1, Q6_P2 = ufl.grad(u1), ufl.grad(u2)
expressions = [(Q6_P1, pts), (Q6_P2, pts)]

v1, v2 = ufl.TestFunction(V1), ufl.TestFunction(V2)
L1 = ufl.inner(1, v1) * ufl.dx
L2 = ufl.inner(1, v2) * ufl.dx
forms = [L1, L2]
