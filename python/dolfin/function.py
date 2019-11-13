# Copyright (C) 2009-2019 Johan Hake, Chris N. Richardson and Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import typing
from functools import singledispatch

import cffi
import numpy as np
from petsc4py import PETSc

import ufl
from dolfin import common, cpp, function, jit
from dolfin.fem import dofmap


class Function(ufl.Coefficient):
    """A finite element function that is represented by a function
    space (domain, element and dofmap) and a vector holding the
    degrees-of-freedom

    """

    def __init__(self,
                 V: "FunctionSpace",
                 x: typing.Optional[PETSc.Vec] = None,
                 name: typing.Optional[str] = None):
        """Initialize finite element Function."""

        # Create cpp Function
        if x is not None:
            self._cpp_object = cpp.function.Function(V._cpp_object, x)
        else:
            self._cpp_object = cpp.function.Function(V._cpp_object)

        # Initialize the ufl.FunctionSpace
        super().__init__(V.ufl_function_space(), count=self._cpp_object.id)

        # Set name
        if name is None:
            self.name = "f_{}".format(self.count())
        else:
            self.name = name

        # Store DOLFIN FunctionSpace object
        self._V = V

    @property
    def function_space(self) -> "FunctionSpace":
        """Return the FunctionSpace"""
        return self._V

    @property
    def value_rank(self) -> int:
        return self._cpp_object.value_rank

    def value_dimension(self, i) -> int:
        return self._cpp_object.value_dimension(i)

    def value_shape(self):
        return self._cpp_object.value_shape

    def ufl_evaluate(self, x, component, derivatives):
        """Function used by ufl to evaluate the Expression"""
        # FIXME: same as dolfin.expression.Expression version. Find way
        # to re-use.
        assert derivatives == ()  # TODO: Handle derivatives

        if component:
            shape = self.ufl_shape
            assert len(shape) == len(component)
            value_size = ufl.product(shape)
            index = ufl.utils.indexflattening.flatten_multiindex(
                component, ufl.utils.indexflattening.shape_to_strides(shape))
            values = np.zeros(value_size)
            # FIXME: use a function with a return value
            self(*x, values=values)
            return values[index]
        else:
            # Scalar evaluation
            return self(*x)

    def eval(self, x: np.ndarray, cells: np.ndarray, u=None) -> np.ndarray:
        """Evaluate Function at points x, where x has shape (num_points, 3),
        and cells has shape (num_points,) and cell[i] is the index of the
        cell containing point x[i]. If the cell index is negative the
        point is ignored."""

        # Make sure input coordinates are a NumPy array
        x = np.asarray(x, dtype=np.float)
        assert x.ndim < 3
        num_points = x.shape[0] if x.ndim == 2 else 1
        x = np.reshape(x, (num_points, -1))
        if x.shape[1] != 3:
            raise ValueError("Coordinate(s) for Function evaluation must have length 3.")

        # Make sure cells are a NumPy array
        cells = np.asarray(cells)
        assert cells.ndim < 2
        num_points_c = cells.shape[0] if cells.ndim == 1 else 1
        cells = np.reshape(cells, num_points_c)

        # Allocate memory for return value if not provided
        if u is None:
            value_size = ufl.product(self.ufl_element().value_shape())
            if common.has_petsc_complex:
                u = np.empty((num_points, value_size), dtype=np.complex128)
            else:
                u = np.empty((num_points, value_size))

        self._cpp_object.eval(x, cells, u)
        if num_points == 1:
            u = np.reshape(u, (-1, ))
        return u

    def interpolate(self, u) -> None:
        """Interpolate an expression"""
        @singledispatch
        def _interpolate(u):
            try:
                self._cpp_object.interpolate(u._cpp_object)
            except AttributeError:
                self._cpp_object.interpolate(u)

        @_interpolate.register(int)
        def _(u_ptr):
            self._cpp_object.interpolate_ptr(u_ptr)

        _interpolate(u)

    def compute_point_values(self):
        return self._cpp_object.compute_point_values()

    def copy(self):
        """Return a copy of the Function. The FunctionSpace is shared and the
        degree-of-freedom vector is copied.

        """
        return function.Function(self.function_space,
                                 self._cpp_object.vector.copy())

    @property
    def vector(self):
        """Return the vector holding Function degrees-of-freedom."""
        return self._cpp_object.vector

    @property
    def name(self) -> str:
        """Name of the Function."""
        return self._cpp_object.name

    @name.setter
    def name(self, name):
        self._cpp_object.name = name

    @property
    def id(self) -> int:
        """Return object id index."""
        return self._cpp_object.id

    def __str__(self):
        """Return a pretty print representation of it self."""
        return self.name

    def sub(self, i: int):
        """Return a sub function.

        The sub functions are numbered from i = 0..N-1, where N is the
        total number of sub spaces.

        """
        return Function(
            self._V.sub(i), self.vector, name="{}-{}".format(str(self), i))

    def split(self):
        """Extract any sub functions.

        A sub function can be extracted from a discrete function that
        is in a mixed, vector, or tensor FunctionSpace. The sub
        function resides in the subspace of the mixed space.

        """
        num_sub_spaces = self.function_space.num_sub_spaces()
        if num_sub_spaces == 1:
            raise RuntimeError("No subfunctions to extract")
        return tuple(self.sub(i) for i in range(num_sub_spaces))

    def collapse(self):
        u_collapsed = self._cpp_object.collapse()
        V_collapsed = function.FunctionSpace(None, self.ufl_element(),
                                             u_collapsed.function_space)
        return Function(V_collapsed, u_collapsed.vector)


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
        """Check whether a FunctionSpace is in this FunctionSpace, or is the
        same as this FunctionSpace.

        """
        return self._cpp_object.contains(V._cpp_object)

    def __contains__(self, u):
        """Check whether a function is in the FunctionSpace."""
        try:
            return u._in(self._cpp_object)
        except AttributeError:
            try:
                return u._cpp_object._in(self._cpp_object)
            except Exception as e:
                raise RuntimeError("Unable to check if object is in FunctionSpace ({})".format(e))

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
