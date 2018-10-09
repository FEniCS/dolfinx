# -*- coding: utf-8 -*-
# Copyright (C) 2017-2018 Chris N. Richardson and Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import ufl
from dolfin import cpp, fem, jit


class FunctionSpace(ufl.FunctionSpace):
    def __init__(self, mesh, element, degree=None, cppV=None):
        """Create a finite element function space."""

        # FIXME: This includes some hacks to support construction of
        # function spaces with existing cpp FunctionSpace, which is
        # typical when extracting subspaces.
        #
        # It would element if an element argument was a single object,
        # e.g. a ufl.FiniteElement or a tuple(family degree)

        if cppV:
            ufl_domain = cppV.mesh().ufl_domain()
            ufl_element = element
            ufl.FunctionSpace.__init__(self, ufl_domain, ufl_element)
            self._cpp_object = cppV
            return

        if isinstance(element, ufl.FiniteElementBase):
            ufl_element = element
        else:
            family = element
            ufl_element = ufl.FiniteElement(
                family, mesh.ufl_cell(), degree, form_degree=None)
        ufl.FunctionSpace.__init__(self, mesh.ufl_domain(), ufl_element)

        # Compile dofmap and element and create DOLFIN objects
        ufc_element, ufc_dofmap = jit.ffc_jit(
            ufl_element,
            form_compiler_parameters=None,
            mpi_comm=mesh.mpi_comm())
        ufc_element = fem.dofmap.make_ufc_finite_element(ufc_element)
        dolfin_element = cpp.fem.FiniteElement(ufc_element)
        dolfin_dofmap = fem.DofMap.fromufc(ufc_dofmap, mesh)

        # Initialize the cpp.FunctionSpace
        self._cpp_object = cpp.function.FunctionSpace(
            mesh, dolfin_element, dolfin_dofmap._cpp_object)

    def dolfin_element(self):
        "Return the DOLFIN element."
        return self._cpp_object.element()

    def num_sub_spaces(self):
        "Return the number of sub spaces"
        return self.dolfin_element().num_sub_elements()

    def sub(self, i: int):
        """Return the i-th sub space"""
        assert self.ufl_element().num_sub_elements() > i
        sub_element = self.ufl_element().sub_elements()[i]
        cppV_sub = self._cpp_object.sub([i])
        return FunctionSpace(None, sub_element, -1, cppV_sub)

    def component(self):
        return self._cpp_object.component()

    def contains(self, V):
        "Check whether a function is in the FunctionSpace"
        return self._cpp_object.contains(V._cpp_object)

    def __contains__(self, u):
        "Check whether a function is in the FunctionSpace"
        try:
            return u._in(self._cpp_object)
        except AttributeError:
            try:
                return u._cpp_object._in(self._cpp_object)
            except Exception as e:
                raise RuntimeError(
                    "Unable to check if object is in FunctionSpace ({})".
                    format(e))

    def __eq__(self, other):
        "Comparison for equality."
        return ufl.FunctionSpace.__eq__(
            self, other) and self._cpp_object == other._cpp_object

    def __ne__(self, other):
        """Comparison for inequality."""
        return ufl.FunctionSpace.__ne__(
            self, other) or self._cpp_object != other._cpp_object

    def ufl_cell(self):
        return self._cpp_object.mesh().ufl_cell()

    def ufl_function_space(self):
        return self

    def dim(self):
        return self._cpp_object.dim

    def id(self):
        return self._cpp_object.id()

    def element(self):
        return self._cpp_object.element()

    def dofmap(self):
        return fem.DofMap(self._cpp_object.dofmap())

    def mesh(self):
        return self._cpp_object.mesh()

    def set_x(self, basis, x, component):
        return self._cpp_object.set_x(basis, x, component)

    def collapse(self, collapsed_dofs: bool = False):
        """Collapse a subspace and return a new function space and a map from
        new to old dofs

        *Arguments*
            collapsed_dofs (bool)
                Return the map from new to old dofs

       *Returns*
           _FunctionSpace_
                The new function space.
           dict
                The map from new to old dofs (optional)

        """
        cpp_space, dofs = self._cpp_object.collapse()
        V = FunctionSpace(None, self.ufl_element(), None, cpp_space)
        if collapsed_dofs:
            return V, dofs
        else:
            return V

    def extract_sub_space(self, component):
        V = self._cpp_object.extract_sub_space(component)
        return FunctionSpace(V)

    def tabulate_dof_coordinates(self):
        return self._cpp_object.tabulate_dof_coordinates()


def VectorFunctionSpace(mesh: cpp.mesh.Mesh,
                        family: str,
                        degree: int,
                        dim=None,
                        form_degree=None,
                        restriction=None):
    """Create vector finite element function space."""

    element = ufl.VectorElement(
        family, mesh.ufl_cell(), degree, form_degree=form_degree, dim=dim)
    return FunctionSpace(mesh, element)


def TensorFunctionSpace(mesh: cpp.mesh.Mesh,
                        family: str,
                        degree: int,
                        shape=None,
                        symmetry=None,
                        restriction=None):
    """Create tensor finite element function space."""

    element = ufl.TensorElement(family, mesh.ufl_cell(), degree, shape,
                                symmetry)
    return FunctionSpace(mesh, element)
