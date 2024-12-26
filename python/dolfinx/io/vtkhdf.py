# Copyright (C) 2024 Chris Richardson
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numpy as np

import basix
import ufl
from dolfinx.cpp.io import read_vtkhdf_mesh_float32, read_vtkhdf_mesh_float64, write_vtkhdf_mesh
from dolfinx.mesh import Mesh


def read_mesh(comm, filename, dtype=np.float64):
    if dtype == np.float64:
        mesh_cpp = read_vtkhdf_mesh_float64(comm, filename)
    elif dtype == np.float32:
        mesh_cpp = read_vtkhdf_mesh_float32(comm, filename)

    cell_types = mesh_cpp.topology.entity_types[-1]
    if len(cell_types) > 1:
        # Not yet defined for mixed topology
        domain = None
    else:
        cell_degree = mesh_cpp.geometry.cmap.degree
        domain = ufl.Mesh(
            basix.ufl.element(
                "Lagrange",
                cell_types[0].name,
                cell_degree,
                basix.LagrangeVariant.equispaced,
                shape=(mesh_cpp.geometry.dim,),
            )
        )
    return Mesh(mesh_cpp, domain)


def write_mesh(filename, mesh):
    write_vtkhdf_mesh(filename, mesh._cpp_object)
