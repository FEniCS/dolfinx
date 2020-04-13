# -*- coding: utf-8 -*-
# Copyright (C) 2018 Michal Habera
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from dolfinx import cpp


class DofMap:
    """Degree-of-freedom map

    This class handles the mapping of degrees of freedom. It builds
    a dof map based on a ufc_dofmap on a specific mesh.
    """

    def __init__(self, dofmap: cpp.fem.DofMap):
        self._cpp_object = dofmap

    def cell_dofs(self, cell_index: int):
        return self._cpp_object.cell_dofs(cell_index)

    @property
    def dof_layout(self):
        return self._cpp_object.dof_layout

    @property
    def index_map(self):
        return self._cpp_object.index_map

    @property
    def list(self):
        return self._cpp_object.list()
