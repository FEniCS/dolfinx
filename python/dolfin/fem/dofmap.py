# -*- coding: utf-8 -*-
# Copyright (C) 2018 Michal Habera
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from dolfin import cpp
from dolfin import fem


def make_ufc_finite_element(ufc_finite_element):
    """Returns ufc finite element from a pointer
    """
    return cpp.fem.make_ufc_finite_element(ufc_finite_element)


def make_ufc_dofmap(ufc_dofmap):
    """Returns ufc dofmap from a pointer
    """
    return cpp.fem.make_ufc_dofmap(ufc_dofmap)


def make_ufc_form(ufc_form):
    """Returns ufc form from a pointer
    """
    return cpp.fem.make_ufc_form(ufc_form)


def make_ufc_coordinate_mapping(ufc_coordinate_mapping):
    """Returns ufc coordinate mapping from a pointer
    """
    return cpp.fem.make_ufc_coordinate_mapping(ufc_coordinate_mapping)


class DofMap:
    """Degree-of-freedom map

    This class handles the mapping of degrees of freedom. It builds
    a dof map based on a ufc_dofmap on a specific mesh.
    """

    def __init__(self, cpp_dofmap=None):
        self._cpp_object = cpp_dofmap

    @classmethod
    def fromcpp(cls, cpp_dofmap):
        """Initialize from C++ dofmap

        Parameters
        ----------
        cpp_dofmap: dolfin.cpp.fem.DofMap
        """
        return cls(cpp_dofmap)

    @classmethod
    def fromufc(cls, ufc_dofmap, mesh):
        """Initialize from UFC dofmap and mesh

        Parameters
        ----------
        ufc_dofmap
            Pointer to ufc_dofmap as returned by FFC JIT
        mesh: dolfin.cpp.mesh.Mesh
        """
        ufc_dofmap = fem.make_ufc_dofmap(ufc_dofmap)
        cpp_dofmap = cpp.fem.DofMap(ufc_dofmap, mesh)
        return cls(cpp_dofmap)

    def global_dimension(self):
        return self._cpp_object.global_dimension()

    def neighbours(self):
        return self._cpp_object.neighbours()

    def shared_nodes(self):
        return self._cpp_object.shared_nodes()

    def ownership_range(self):
        return self._cpp_object.ownership_range()

    def cell_dofs(self, cell_index: int):
        return self._cpp_object.cell_dofs(cell_index)

    def dofs(self, mesh, entity_dim: int):
        return self._cpp_object.dofs(mesh, entity_dim)

    def entity_dofs_all(self, mesh, entity_dim: int):
        return self._cpp_object.entity_dofs(mesh, entity_dim)

    def entity_dofs(self, mesh, entity_dim: int, entity_index: int):
        return self._cpp_object.entity_dofs(mesh, entity_dim, entity_index)

    def num_entity_dofs(self, entity_dim: int):
        return self._cpp_object.num_entity_dofs(entity_dim)

    def tabulate_local_to_global_dofs(self):
        return self._cpp_object.tabulate_local_to_global_dofs()

    def tabulate_entity_dofs(self, entity_dim: int, cell_entity_index: int):
        return self._cpp_object.tabulate_entity_dofs(entity_dim,
                                                     cell_entity_index)

    def block_size(self):
        return self._cpp_object.block_size()

    def set(self, petscvec, value):
        self._cpp_object.set(petscvec, value)

    def index_map(self):
        return self._cpp_object.index_map()
