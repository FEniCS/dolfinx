# Copyright (C) 2021 Joseph P. Dean
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

# FIXME This seems a bit of a hack
import ufl
import dolfinx


def ufl_cell(self):
    return ufl.Cell(self.topology.cell_name(),
                    geometric_dimension=self.parent_mesh.geometry.dim)


def ufl_domain(self):
    # TODO Get degree from mesh
    return ufl.Mesh(ufl.VectorElement("Lagrange", cell=self.ufl_cell(),
                                      degree=1,
                                      dim=self.parent_mesh.geometry.dim))


dolfinx.cpp.mesh.MeshView.ufl_cell = ufl_cell
dolfinx.cpp.mesh.MeshView.ufl_domain = ufl_domain
