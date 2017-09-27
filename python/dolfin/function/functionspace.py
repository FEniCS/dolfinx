# -*- coding: utf-8 -*-
"""Main module for DOLFIN"""

# Copyright (C) 2017 Chris N. Richardson and Garth N. Wells
#
# Distributed under the terms of the GNU Lesser Public License (LGPL),
# either version 3 of the License, or (at your option) any later
# version.

import ufl
import dolfin.cpp as cpp
from dolfin.jit.jit import ffc_jit
from . import function


class FunctionSpace(ufl.FunctionSpace):

    def __init__(self, *args, **kwargs):
        """Create finite element function space."""

        if len(args) == 1:
            # Do we relly want to do it this way? Can we get the
            # sub-element from UFL?
            self._init_from_cpp(*args, **kwargs)
        else:
            if len(args) == 0 or not isinstance(args[0], cpp.mesh.Mesh):
                # cpp.dolfin_error("functionspace.py",
                #                  "create function space",
                #                  "Illegal argument, not a mesh: "
                #                  + str(args[0]))
                pass
            elif len(args) == 2:
                self._init_from_ufl(*args, **kwargs)
            else:
                self._init_convenience(*args, **kwargs)

    def _init_from_ufl(self, mesh, element, constrained_domain=None):

        # Initialize the ufl.FunctionSpace first to check for good
        # meaning
        ufl.FunctionSpace.__init__(self, mesh.ufl_domain(), element)

        # Compile dofmap and element
        ufc_element, ufc_dofmap = ffc_jit(element, form_compiler_parameters=None,
                                          mpi_comm=mesh.mpi_comm())
        ufc_element = cpp.fem.make_ufc_finite_element(ufc_element)

        # Create DOLFIN element and dofmap
        dolfin_element = cpp.fem.FiniteElement(ufc_element)
        ufc_dofmap = cpp.fem.make_ufc_dofmap(ufc_dofmap)
        if constrained_domain is None:
            dolfin_dofmap = cpp.fem.DofMap(ufc_dofmap, mesh)
        else:
            dolfin_dofmap = cpp.fem.DofMap(ufc_dofmap, mesh,
                                           constrained_domain)

        # Initialize the cpp.FunctionSpace
        self._cpp_object = cpp.function.FunctionSpace(mesh,
                                                      dolfin_element,
                                                      dolfin_dofmap)

    def _init_from_cpp(self, cppV, **kwargs):
        """
        if not isinstance(cppV, cpp.FunctionSpace):
            cpp.dolfin_error("functionspace.py",
                             "create function space",
                             "Illegal argument for C++ function space, "
                             "not a cpp.FunctionSpace: " + str(cppV))
        # We don't want to support copy construction. This would
        # indicate internal defficiency in the library
        if isinstance(cppV, FunctionSpace):
            cpp.dolfin_error("functionspace.py",
                             "create function space",
                             "Illegal argument for C++ function space, "
                             "should not be functions.functionspace.FunctionSpace: " + str(cppV))
        if len(kwargs) > 0:
            cpp.dolfin_error("functionspace.py",
                             "create function space",
                             "Illegal arguments, did not expect C++ "
                             "function space and **kwargs: " + str(kwargs))
        """

        # Reconstruct UFL element from signature
        ufl_element = eval(cppV.element().signature(), ufl.__dict__)

        # Get mesh
        ufl_domain = cppV.mesh().ufl_domain()

        # Initialize the ufl.FunctionSpace (not calling cpp.Function.__init__)
        self._cpp_object = cppV

        # Initialize the ufl.FunctionSpace
        ufl.FunctionSpace.__init__(self, ufl_domain, ufl_element)

    def _init_convenience(self, mesh, family, degree, form_degree=None,
                          constrained_domain=None, restriction=None):

        # Create UFL element
        element = ufl.FiniteElement(family, mesh.ufl_cell(), degree,
                                    form_degree=form_degree)

        self._init_from_ufl(mesh, element, constrained_domain=constrained_domain)

    def dolfin_element(self):
        "Return the DOLFIN element."
        return self._cpp_object.element()

    def num_sub_spaces(self):
        "Return the number of sub spaces"
        return self.dolfin_element().num_sub_elements()

    def sub(self, i):
        "Return the i-th sub space"
        # FIXME: Should we have a more extensive check other than
        # whats includeding the cpp code?
        if not isinstance(i, int):
            raise TypeError("expected an int for 'i'")
        if self.num_sub_spaces() == 1:
            raise ValueError("no SubSpaces to extract")
        if i >= self.num_sub_spaces():
            raise ValueError("Can only extract SubSpaces with i = 0 ... %d" %
                             (self.num_sub_spaces() - 1))
        assert hasattr(self.ufl_element(), "sub_elements")

        # Extend with the python layer
        return FunctionSpace(cpp.function.FunctionSpace.sub(self._cpp_object, i))

    def component(self):
        return self._cpp_object.component()

    def contains(self, V):
        "Check whether a function is in the FunctionSpace"
        return self._cpp_object.contains(V._cpp_object)
        # if isinstance(u, cpp.function.Function):
        #    return u._in(self)
        # elif isinstance(u, function.Function):
        #    return u._cpp_object._in(self)
        # return False

    def __contains__(self, u):
        "Check whether a function is in the FunctionSpace"
        if isinstance(u, cpp.function.Function):
            return u._in(self._cpp_object)
        elif isinstance(u, function.Function):
            return u._cpp_object._in(self._cpp_object)
        return False

    def __eq__(self, other):
        "Comparison for equality."
        return ufl.FunctionSpace.__eq__(self, other) and self._cpp_object == other._cpp_object

    def __ne__(self, other):
        "Comparison for inequality."
        return ufl.FunctionSpace.__ne__(self, other) or self._cpp_object != other._cpp_object

    def ufl_cell(self):
        return self._cpp_object.mesh().ufl_cell()

    def ufl_function_space(self):
        return self

    def dim(self):
        return self._cpp_object.dim()

    def id(self):
        return self._cpp_object.id()

    def element(self):
        return self._cpp_object.element()

    def dofmap(self):
        return self._cpp_object.dofmap()

    def mesh(self):
        return self._cpp_object.mesh()

    def set_x(self, basis, x, component):
        return self._cpp_object.set_x(basis, x, component)

    def collapse(self, collapsed_dofs=False):
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
        # Get the cpp version of the FunctionSpace
        cpp_space, dofs = self._cpp_object.collapse()

        # Extend with the python layer
        V = FunctionSpace(cpp_space)

        if collapsed_dofs:
            return V, dofs
        else:
            return V

    def extract_sub_space(self, component):
        V = self._cpp_object.extract_sub_space(component)
        return FunctionSpace(V)

    def tabulate_dof_coordinates(self):
        return self._cpp_object.tabulate_dof_coordinates()


def VectorFunctionSpace(mesh, family, degree, dim=None, form_degree=None,
                        constrained_domain=None, restriction=None):
    """Create finite element function space."""

    # Create UFL element
    element = ufl.VectorElement(family, mesh.ufl_cell(), degree,
                                form_degree=form_degree, dim=dim)

    # Return (Py)DOLFIN FunctionSpace
    return FunctionSpace(mesh, element, constrained_domain=constrained_domain)


def TensorFunctionSpace(mesh, family, degree, shape=None, symmetry=None,
                        constrained_domain=None, restriction=None):
    """Create finite element function space."""

    # Create UFL element
    element = ufl.TensorElement(family, mesh.ufl_cell(), degree,
                                shape, symmetry)

    # Return (Py)DOLFIN FunctionSpace
    return FunctionSpace(mesh, element, constrained_domain=constrained_domain)
