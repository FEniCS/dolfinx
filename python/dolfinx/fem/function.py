# Copyright (C) 2009-2019 Chris N. Richardson, Garth N. Wells and Michal Habera
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Collection of functions and function spaces"""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from dolfinx.mesh import Mesh

from functools import singledispatch

import cffi
import numpy as np

import ufl
import ufl.algorithms
import ufl.algorithms.analysis
from dolfinx import cpp as _cpp
from dolfinx import jit
from dolfinx.fem import dofmap

from petsc4py import PETSc


class Constant(ufl.Constant):
    def __init__(self, domain, c: typing.Union[np.ndarray, typing.Sequence, float]):
        """A constant with respect to a domain.

        Parameters
        ----------
        domain : DOLFINx or UFL mesh
        c
            Value of the constant.
        """
        c_np = np.asarray(c)
        super().__init__(domain, c_np.shape)
        if c_np.dtype == np.complex64:
            self._cpp_object = _cpp.fem.Constant_complex64(c_np)
        elif c_np.dtype == np.complex128:
            self._cpp_object = _cpp.fem.Constant_complex128(c_np)
        elif c_np.dtype == np.float32:
            self._cpp_object = _cpp.fem.Constant_float32(c_np)
        elif c_np.dtype == np.float64:
            self._cpp_object = _cpp.fem.Constant_float64(c_np)
        else:
            raise RuntimeError("Unsupported dtype")

    @property
    def value(self):
        """The value of the constant"""
        return self._cpp_object.value

    @value.setter
    def value(self, v):
        np.copyto(self._cpp_object.value, np.asarray(v))

    @property
    def dtype(self):
        return self.value.dtype


class Expression:
    def __init__(self, ufl_expression: ufl.core.expr.Expr, x: np.ndarray,
                 form_compiler_parameters: dict = {}, jit_parameters: dict = {},
                 dtype=PETSc.ScalarType):
        """Create DOLFINx Expression.

        Represents a mathematical expression evaluated at a pre-defined set of
        points on the reference cell. This class closely follows the concept of a
        UFC Expression.

        This functionality can be used to evaluate a gradient of a Function at
        the quadrature points in all cells. This evaluated gradient can then be
        used as input to a non-FEniCS function that calculates a material
        constitutive model.

        Parameters
        ----------
        ufl_expression
            Pure UFL expression
        x
            Array of points of shape (num_points, tdim) on the reference
            element.
        form_compiler_parameters
            Parameters used in FFCx compilation of this Expression. Run `ffcx
            --help` in the commandline to see all available options.
        jit_parameters
            Parameters controlling JIT compilation of C code.

        Notes
        -----
        This wrapper is responsible for the FFCx compilation of the UFL Expr
        and attaching the correct data to the underlying C++ Expression.

        """

        assert x.ndim < 3
        num_points = x.shape[0] if x.ndim == 2 else 1
        x = np.reshape(x, (num_points, -1))

        mesh = ufl_expression.ufl_domain().ufl_cargo()

        # Compile UFL expression with JIT
        if dtype == np.float64:
            form_compiler_parameters["scalar_type"] = "double"
        elif dtype == np.complex128:
            form_compiler_parameters["scalar_type"] = "double _Complex"
        else:
            raise RuntimeError(f"Unsupported scalar type {dtype} for Form.")
        self._ufcx_expression, module, self._code = jit.ffcx_jit(mesh.comm, (ufl_expression, x),
                                                                 form_compiler_parameters=form_compiler_parameters,
                                                                 jit_parameters=jit_parameters)
        self._ufl_expression = ufl_expression

        # Setup data (evaluation points, coefficients, constants, mesh, value_size).
        # Tabulation function.
        ffi = cffi.FFI()
        fn = ffi.cast("uintptr_t", self.ufcx_expression.tabulate_expression)

        value_size = ufl.product(self.ufl_expression.ufl_shape)

        ufl_coefficients = ufl.algorithms.extract_coefficients(ufl_expression)
        coefficients = [ufl_coefficient._cpp_object for ufl_coefficient in ufl_coefficients]

        ufl_constants = ufl.algorithms.analysis.extract_constants(ufl_expression)
        constants = [ufl_constant._cpp_object for ufl_constant in ufl_constants]

        # Getcpp Expression type
        def expressiontype(dtype):
            if dtype is np.float64:
                return _cpp.fem.Expression_float64
            elif dtype is np.complex128:
                return _cpp.fem.Expression_complex128
            else:
                raise NotImplementedError(f"Type {dtype} not supported.")

        self._cpp_object = expressiontype(dtype)(coefficients, constants, mesh, x, fn, value_size)

    def eval(self, cells: np.ndarray, u: typing.Optional[np.ndarray] = None) -> np.ndarray:
        """Evaluate Expression in cells.

        Parameters
        ----------
        cells
            local indices of cells to evaluate expression.
        u: optional
            array of shape (num_cells, num_points*value_size) to
            store result of expression evaluation.

        Returns
        -------

        u: np.ndarray
            The i-th row of u contains the expression evaluated on cells[i].

        Note
        ----
        This function allocates u of the appropriate size if u is not passed.
        """
        cells = np.asarray(cells, dtype=np.int32)
        assert cells.ndim == 1
        num_cells = cells.shape[0]

        # Allocate memory for result if u was not provided
        if u is None:
            if np.issubdtype(PETSc.ScalarType, np.complexfloating):
                u = np.empty((num_cells, self.x.shape[0] * self.value_size), dtype=np.complex128)
            else:
                u = np.empty((num_cells, self.x.shape[0] * self.value_size), dtype=np.float64)
            self._cpp_object.eval(cells, u)
        else:
            self._cpp_object.eval(cells, u)

        return u

    @property
    def ufl_expression(self):
        """Original UFL Expression"""
        return self._ufl_expression

    @property
    def x(self) -> np.ndarray:
        """Evaluation points on the reference cell"""
        return self._cpp_object.x

    @property
    def value_size(self) -> int:
        """Value size of the expression"""
        return self._cpp_object.value_size

    @property
    def ufcx_expression(self):
        """The compiled ufcx_expression object"""
        return self._ufcx_expression

    @property
    def code(self) -> str:
        """C code strings"""
        return self._code


class Function(ufl.Coefficient):
    """A finite element function that is represented by a function space
    (domain, element and dofmap) and a vector holding the
    degrees-of-freedom

    """

    def __init__(self,
                 V: FunctionSpace,
                 x: typing.Optional[typing.Union[_cpp.la.Vector_float64, _cpp.la.Vector_complex128]] = None,
                 name: typing.Optional[str] = None,
                 dtype=PETSc.ScalarType):
        """Initialize finite element Function."""

        # Create cpp Function
        def functiontype(dtype):
            if dtype is np.float64:
                return _cpp.fem.Function_float64
            elif dtype is np.complex128:
                return _cpp.fem.Function_complex128
            else:
                raise NotImplementedError(f"Type {dtype} not supported.")

        if x is not None:
            self._cpp_object = functiontype(dtype)(V._cpp_object, x)
        else:
            self._cpp_object = functiontype(dtype)(V._cpp_object)

        # Initialize the ufl.FunctionSpace
        super().__init__(V.ufl_function_space(), count=self._cpp_object.id)

        # Set name
        if name is None:
            self.name = "f_{}".format(self.count())
        else:
            self.name = name

        # Store DOLFINx FunctionSpace object
        self._V = V

        # PETSc Vec wrapper around the C++ function data. Constructed
        # when first requested.
        self._petsc_x = None

    @property
    def function_space(self) -> FunctionSpace:
        """The FunctionSpace that the Function is defined on"""
        return self._V

    def ufl_evaluate(self, x, component, derivatives):
        """Function used by ufl to evaluate the Expression"""
        # FIXME: same as dolfinx.expression.Expression version. Find way
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
        x = np.asarray(x, dtype=np.float64)
        assert x.ndim < 3
        num_points = x.shape[0] if x.ndim == 2 else 1
        x = np.reshape(x, (num_points, -1))
        if x.shape[1] != 3:
            raise ValueError("Coordinate(s) for Function evaluation must have length 3.")

        # Make sure cells are a NumPy array
        cells = np.asarray(cells, dtype=np.int32)
        assert cells.ndim < 2
        num_points_c = cells.shape[0] if cells.ndim == 1 else 1
        cells = np.reshape(cells, num_points_c)

        # Allocate memory for return value if not provided
        if u is None:
            value_size = ufl.product(self.ufl_element().value_shape())
            if np.issubdtype(PETSc.ScalarType, np.complexfloating):
                u = np.empty((num_points, value_size), dtype=np.complex128)
            else:
                u = np.empty((num_points, value_size))

        self._cpp_object.eval(x, cells, u)
        if num_points == 1:
            u = np.reshape(u, (-1, ))
        return u

    def interpolate(self, u: typing.Union[typing.Callable, Expression, Function],
                    cells: np.ndarray = None) -> None:
        """Interpolate an expression

        Args:
            u: The function, Expression or Function to interpolate
            cells: The cells to interpolate over. If `None` then all
                cells are interpolated over

        """
        @singledispatch
        def _interpolate(u, cells):
            try:
                self._cpp_object.interpolate(u._cpp_object, cells)
            except AttributeError:
                self._cpp_object.interpolate(u, cells)

        @_interpolate.register(int)
        def _(u_ptr, cells):
            self._cpp_object.interpolate_ptr(u_ptr, cells)

        if cells is None:
            mesh = self.function_space.mesh
            map = mesh.topology.index_map(mesh.topology.dim)
            cells = np.arange(map.size_local + map.num_ghosts, dtype=np.int32)
        _interpolate(u, cells)

    def compute_point_values(self):
        return self._cpp_object.compute_point_values()

    def copy(self) -> Function:
        """Return a copy of the Function. The FunctionSpace is shared and the
        degree-of-freedom vector is copied.

        """
        return Function(self.function_space, type(self.x)(self.x))

    @property
    def vector(self):
        """PETSc vector holding the degrees-of-freedom."""
        if self._petsc_x is None:
            self._petsc_x = _cpp.la.petsc.create_vector_wrap(self.x)
        return self._petsc_x

    @property
    def x(self):
        """Vector holding the degrees-of-freedom."""
        return self._cpp_object.x

    @property
    def dtype(self) -> np.dtype:
        return self._cpp_object.x.array.dtype

    @property
    def name(self) -> str:
        """Name of the Function."""
        return self._cpp_object.name

    @name.setter
    def name(self, name):
        self._cpp_object.name = name

    @property
    def id(self) -> int:
        """Pbject id index."""
        return self._cpp_object.id

    def __str__(self):
        """Pretty print representation of it self."""
        return self.name

    def sub(self, i: int):
        """Return a sub function.

        The sub functions are numbered from i = 0..N-1, where N is the
        total number of sub spaces.

        """
        return Function(self._V.sub(i), self.x, name="{}-{}".format(str(self), i))

    def split(self):
        """Extract any sub functions.

        A sub function can be extracted from a discrete function that
        is in a mixed, vector, or tensor FunctionSpace. The sub
        function resides in the subspace of the mixed space.

        """
        num_sub_spaces = self.function_space.num_sub_spaces
        if num_sub_spaces == 1:
            raise RuntimeError("No subfunctions to extract")
        return tuple(self.sub(i) for i in range(num_sub_spaces))

    def collapse(self) -> Function:
        u_collapsed = self._cpp_object.collapse()
        V_collapsed = FunctionSpace(None, self.ufl_element(),
                                    u_collapsed.function_space)
        return Function(V_collapsed, u_collapsed.x)


class ElementMetaData(typing.NamedTuple):
    """Data for representing a finite element"""
    family: str
    degree: int
    form_degree: typing.Optional[int] = None  # noqa


class FunctionSpace(ufl.FunctionSpace):
    """A space on which Functions (fields) can be defined."""

    def __init__(self,
                 mesh: Mesh,
                 element: typing.Union[ufl.FiniteElementBase, ElementMetaData],
                 cppV: typing.Optional[_cpp.fem.FunctionSpace] = None,
                 form_compiler_parameters: dict = {},
                 jit_parameters: dict = {}):
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
        (self._ufcx_element, self._ufcx_dofmap), module, code = jit.ffcx_jit(
            mesh.comm, self.ufl_element(), form_compiler_parameters=form_compiler_parameters,
            jit_parameters=jit_parameters)

        ffi = cffi.FFI()
        cpp_element = _cpp.fem.FiniteElement(ffi.cast("uintptr_t", ffi.addressof(self._ufcx_element)))
        cpp_dofmap = _cpp.fem.create_dofmap(mesh.comm, ffi.cast(
            "uintptr_t", ffi.addressof(self._ufcx_dofmap)), mesh.topology, cpp_element)

        # Initialize the cpp.FunctionSpace
        self._cpp_object = _cpp.fem.FunctionSpace(mesh, cpp_element, cpp_dofmap)

    def clone(self) -> FunctionSpace:
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
        Vcpp = _cpp.fem.FunctionSpace(self._cpp_object.mesh, self._cpp_object.element, self._cpp_object.dofmap)
        return FunctionSpace(None, self.ufl_element(), Vcpp)

    @property
    def num_sub_spaces(self) -> int:
        """Number of sub spaces"""
        return self.element.num_sub_elements

    def sub(self, i: int) -> FunctionSpace:
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
        """UFL function space"""
        return self

    @property
    def id(self) -> int:
        """Unique identifier"""
        return self._cpp_object.id

    @property
    def element(self):
        return self._cpp_object.element

    @property
    def dofmap(self) -> dofmap.DofMap:
        """Degree-of-freedom map associated with the function space."""
        return dofmap.DofMap(self._cpp_object.dofmap)

    @property
    def mesh(self) -> _cpp.mesh.Mesh:
        """Return the mesh on which the function space is defined."""
        return self._cpp_object.mesh

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

    def tabulate_dof_coordinates(self) -> np.ndarray:
        return self._cpp_object.tabulate_dof_coordinates()


def VectorFunctionSpace(mesh: Mesh, element: ElementMetaData, dim=None,
                        restriction=None) -> FunctionSpace:
    """Create vector finite element (composition of scalar elements) function space."""

    e = ElementMetaData(*element)
    ufl_element = ufl.VectorElement(e.family, mesh.ufl_cell(), e.degree, form_degree=e.form_degree, dim=dim)
    return FunctionSpace(mesh, ufl_element)


def TensorFunctionSpace(mesh: Mesh, element: ElementMetaData, shape=None,
                        symmetry: bool = None, restriction=None) -> FunctionSpace:
    """Create tensor finite element (composition of scalar elements) function space."""

    e = ElementMetaData(*element)
    ufl_element = ufl.TensorElement(e.family, mesh.ufl_cell(), e.degree, shape, symmetry)
    return FunctionSpace(mesh, ufl_element)
