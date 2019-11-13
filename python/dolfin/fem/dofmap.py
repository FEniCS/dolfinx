# -*- coding: utf-8 -*-
# Copyright (C) 2018 Michal Habera
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from dolfinx import cpp


def make_ufc_finite_element(ufc_finite_element):
    """Returns ufc finite element from a pointer"""
    return cpp.fem.make_ufc_finite_element(ufc_finite_element)


def make_ufc_dofmap(ufc_dofmap):
    """Returns ufc dofmap from a pointer"""
    return cpp.fem.make_ufc_dofmap(ufc_dofmap)


def make_ufc_form(ufc_form):
    """Returns ufc form from a pointer"""
    return cpp.fem.make_ufc_form(ufc_form)


def make_coordinate_mapping(ufc_coordinate_mapping):
    """Returns CoordinateElement from a pointer to a ufc_coordinate_mapping"""
    return cpp.fem.make_coordinate_mapping(ufc_coordinate_mapping)


class DofMap:
    """Degree-of-freedom map

    This class handles the mapping of degrees of freedom. It builds
    a dof map based on a ufc_dofmap on a specific mesh.
    """

    def __init__(self, dofmap: cpp.fem.DofMap):
        self._cpp_object = dofmap

    def cell_dofs(self, cell_index: int):
        return self._cpp_object.cell_dofs(cell_index)

    def dofs(self, mesh, entity_dim: int):
        return self._cpp_object.dofs(mesh, entity_dim)

    def set(self, x, value):
        self._cpp_object.set(x, value)

    @property
    def dof_layout(self):
        return self._cpp_object.dof_layout

    @property
    def index_map(self):
        return self._cpp_object.index_map

    @property
    def dof_array(self):
        return self._cpp_object.dof_array()
