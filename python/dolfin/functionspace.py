# Copyright (C) 2017-2018 Chris N. Richardson and Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import typing

import cffi

import ufl
from dolfin import cpp, jit
from dolfin.fem import dofmap


class ElementMetaData(typing.NamedTuple):
    """Data for representing a finite element"""
    family: str
    degree: int
    form_degree: typing.Optional[int] = None  # noqa


class FunctionSpace(ufl.FunctionSpace):
    """A space on which Functions (fields) can be defined."""

    def __init__(self,
                 mesh: cpp.mesh.Mesh,
                 element: typing.Union[ufl.FiniteElementBase, ElementMetaData],
                 cppV: typing.Optional[cpp.function.FunctionSpace] = None):
        """Create a finite element function space."""

        # Create function space from a UFL element and existing cpp
        # FunctionSpace
        if cppV is not None:
            assert mesh is None
            ufl_domain = cppV.mesh.ufl_domain()
            super().__init__(ufl_domain, element)
            self._cpp_object = cppV
            return

        # Initialise the ufl.FunctionSpace
        if isinstance(element, ufl.FiniteElementBase):
            super().__init__(mesh.ufl_domain(), element)
        else:
            e = ElementMetaData(*element)
            ufl_element = ufl.FiniteElement(e.family, mesh.ufl_cell(), e.degree, form_degree=e.form_degree)
            super().__init__(mesh.ufl_domain(), ufl_element)

        # Compile dofmap and element and create DOLFIN objects
        ufc_element, ufc_dofmap_ptr = jit.ffc_jit(
            self.ufl_element(), form_compiler_parameters=None, mpi_comm=mesh.mpi_comm())

        ffi = cffi.FFI()
        ufc_element = dofmap.make_ufc_finite_element(ffi.cast("uintptr_t", ufc_element))
        cpp_element = cpp.fem.FiniteElement(ufc_element)

        ufc_dofmap = dofmap.make_ufc_dofmap(ffi.cast("uintptr_t", ufc_dofmap_ptr))
        cpp_dofmap = cpp.fem.create_dofmap(ufc_dofmap, mesh)

        # Initialize the cpp.FunctionSpace
        self._cpp_object = cpp.function.FunctionSpace(mesh, cpp_element, cpp_dofmap)

    def clone(self) -> "FunctionSpace":
        """Return a new FunctionSpace :math:`W` which shares data with this
        FunctionSpace :math:`V`, but with a different unique integer ID.

        This function is helpful for defining mixed problems and using
        blocked linear algebra. For example, a matrix block defined on
        the spaces :math:`V \\times W` where, :math:`V` and :math:`W`
        are defined on the same finite element and mesh can be
        identified as an off-diagonal block whereas the :math:`V \\times
        V` and :math:`V \\times V` matrices can be identified as
        diagonal blocks. This is relevant for the handling of boundary
        conditions.

        """
        Vcpp = cpp.function.FunctionSpace(self._cpp_object.mesh, self._cpp_object.element, self._cpp_object.dofmap)
        return FunctionSpace(None, self.ufl_element(), Vcpp)

    def dolfin_element(self):
        """Return the DOLFIN element."""
        return self._cpp_object.element

    def num_sub_spaces(self) -> int:
        """Return the number of sub spaces."""
        return self.dolfin_element().num_sub_elements()

    def sub(self, i: int) -> "FunctionSpace":
        """Return the i-th sub space."""
        assert self.ufl_element().num_sub_elements() > i
        sub_element = self.ufl_element().sub_elements()[i]
        cppV_sub = self._cpp_object.sub([i])
        return FunctionSpace(None, sub_element, cppV_sub)

    def component(self):
        """Return the component relative to the parent space."""
        return self._cpp_object.component()

    def contains(self, V) -> bool:
        """Check whether a function is in the FunctionSpace."""
        return self._cpp_object.contains(V._cpp_object)

    def __contains__(self, u):
        """Check whether a function is in the FunctionSpace."""
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
        """Comparison for equality."""
        return super().__eq__(other) and self._cpp_object == other._cpp_object

    def __ne__(self, other):
        """Comparison for inequality."""
        return super().__ne__(other) or self._cpp_object != other._cpp_object

    def ufl_cell(self):
        return self._cpp_object.mesh.ufl_cell()

    def ufl_function_space(self) -> ufl.FunctionSpace:
        """Return the UFL function space"""
        return self

    @property
    def dim(self) -> int:
        return self._cpp_object.dim

    @property
    def id(self) -> int:
        """The unique identifier"""
        return self._cpp_object.id

    @property
    def element(self):
        return self._cpp_object.element

    @property
    def dofmap(self) -> dofmap.DofMap:
        """Return the degree-of-freedom map associated with the function space."""
        return dofmap.DofMap(self._cpp_object.dofmap)

    @property
    def mesh(self):
        """Return the mesh on which the function space is defined."""
        return self._cpp_object.mesh

    def set_x(self, basis, x, component) -> None:
        return self._cpp_object.set_x(basis, x, component)

    def collapse(self, collapsed_dofs: bool = False):
        """Collapse a subspace and return a new function space and a map from
        new to old dofs.

        *Arguments*
            collapsed_dofs
                Return the map from new to old dofs

       *Returns*
           FunctionSpace
                The new function space.
           dict
                The map from new to old dofs (optional)

        """
        cpp_space, dofs = self._cpp_object.collapse()
        V = FunctionSpace(None, self.ufl_element(), cpp_space)
        if collapsed_dofs:
            return V, dofs
        else:
            return V

    def tabulate_dof_coordinates(self):
        return self._cpp_object.tabulate_dof_coordinates()


def VectorFunctionSpace(mesh: cpp.mesh.Mesh,
                        element: ElementMetaData,
                        dim=None,
                        restriction=None) -> "FunctionSpace":
    """Create vector finite element (composition of scalar elements) function space."""

    e = ElementMetaData(*element)
    ufl_element = ufl.VectorElement(e.family, mesh.ufl_cell(), e.degree, form_degree=e.form_degree, dim=dim)
    return FunctionSpace(mesh, ufl_element)


def TensorFunctionSpace(mesh: cpp.mesh.Mesh,
                        element: ElementMetaData,
                        shape=None,
                        symmetry: bool = None,
                        restriction=None) -> "FunctionSpace":
    """Create tensor finite element (composition of scalar elements) function space."""

    e = ElementMetaData(*element)
    ufl_element = ufl.TensorElement(e.family, mesh.ufl_cell(), e.degree, shape, symmetry)
    return FunctionSpace(mesh, ufl_element)
