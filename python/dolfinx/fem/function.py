# Copyright (C) 2009-2024 Chris N. Richardson, Garth N. Wells,
# Michal Habera and Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Finite element function spaces and functions."""

from __future__ import annotations

import typing
from collections.abc import Callable, Sequence
from functools import cached_property, singledispatch

import numpy as np
import numpy.typing as npt

import basix
import ufl
from dolfinx import cpp as _cpp
from dolfinx import default_scalar_type, jit, la
from dolfinx.fem import dofmap
from dolfinx.fem.element import FiniteElement, finiteelement
from dolfinx.geometry import PointOwnershipData

if typing.TYPE_CHECKING:
    from mpi4py import MPI as _MPI

    from dolfinx.mesh import Mesh


class Constant(ufl.Constant):
    _cpp_object: typing.Union[
        _cpp.fem.Constant_complex64,
        _cpp.fem.Constant_complex128,
        _cpp.fem.Constant_float32,
        _cpp.fem.Constant_float64,
    ]

    def __init__(
        self, domain, c: typing.Union[np.ndarray, Sequence, np.floating, np.complexfloating]
    ):
        """A constant with respect to a domain.

        Args:
            domain: DOLFINx or UFL mesh
            c: Value of the constant.
        """
        c = np.asarray(c)
        super().__init__(domain, c.shape)
        try:
            if np.issubdtype(c.dtype, np.complex64):
                self._cpp_object = _cpp.fem.Constant_complex64(c)
            elif np.issubdtype(c.dtype, np.complex128):
                self._cpp_object = _cpp.fem.Constant_complex128(c)
            elif np.issubdtype(c.dtype, np.float32):
                self._cpp_object = _cpp.fem.Constant_float32(c)
            elif np.issubdtype(c.dtype, np.float64):
                self._cpp_object = _cpp.fem.Constant_float64(c)
            else:
                raise RuntimeError("Unsupported dtype")
        except AttributeError:
            raise AttributeError("Constant value must have a dtype attribute.")

    @property
    def value(self):
        """The value of the constant"""
        return self._cpp_object.value

    @value.setter
    def value(self, v):
        np.copyto(self._cpp_object.value, np.asarray(v))

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self._cpp_object.dtype)

    def __float__(self):
        if self.ufl_shape or self.ufl_free_indices:
            raise TypeError("Cannot evaluate a nonscalar expression to a scalar value.")
        else:
            return float(self.value)

    def __complex__(self):
        if self.ufl_shape or self.ufl_free_indices:
            raise TypeError("Cannot evaluate a nonscalar expression to a scalar value.")
        else:
            return complex(self.value)


class Expression:
    def __init__(
        self,
        e: ufl.core.expr.Expr,
        X: np.ndarray,
        comm: typing.Optional[_MPI.Comm] = None,
        form_compiler_options: typing.Optional[dict] = None,
        jit_options: typing.Optional[dict] = None,
        dtype: typing.Optional[npt.DTypeLike] = None,
    ):
        """Create a DOLFINx Expression.

        Represents a mathematical expression evaluated at a pre-defined
        set of points on the reference cell. This class closely follows
        the concept of a UFC Expression.

        This functionality can be used to evaluate a gradient of a
        Function at the quadrature points in all cells. This evaluated
        gradient can then be used as input to a non-FEniCS function that
        calculates a material constitutive model.

        Args:
            e: UFL expression.
            X: Array of points of shape ``(num_points, tdim)`` on the
                reference element.
            comm: Communicator that the Expression is defined on.
            form_compiler_options: Options used in FFCx compilation of
                this Expression. Run ``ffcx --help`` in the commandline
                to see all available options.
            jit_options: Options controlling JIT compilation of C code.

        Note:
            This wrapper is responsible for the FFCx compilation of the
            UFL Expr and attaching the correct data to the underlying
            C++ Expression.
        """
        assert X.ndim < 3
        num_points = X.shape[0] if X.ndim == 2 else 1
        _X = np.reshape(X, (num_points, -1))

        # Get MPI communicator
        if comm is None:
            try:
                mesh = ufl.domain.extract_unique_domain(e).ufl_cargo()
                comm = mesh.comm
            except AttributeError:
                print(
                    "Could not extract MPI communicator for Expression. "
                    + "Maybe you need to pass a communicator?"
                )
                raise

        # Attempt to deduce dtype
        if dtype is None:
            dtype = getattr(e, "dtype", default_scalar_type)

        # Compile UFL expression with JIT
        if form_compiler_options is None:
            form_compiler_options = dict()
        form_compiler_options["scalar_type"] = dtype
        self._ufcx_expression, module, self._code = jit.ffcx_jit(
            comm, (e, _X), form_compiler_options=form_compiler_options, jit_options=jit_options
        )
        self._ufl_expression = e

        # Prepare coefficients data. For every coefficient in expression
        # take its C++ object.
        original_coefficients = ufl.algorithms.extract_coefficients(e)
        coeffs = [
            original_coefficients[
                self._ufcx_expression.original_coefficient_positions[i]
            ]._cpp_object
            for i in range(self._ufcx_expression.num_coefficients)
        ]
        ufl_constants = ufl.algorithms.analysis.extract_constants(e)
        constants = [constant._cpp_object for constant in ufl_constants]
        arguments = ufl.algorithms.extract_arguments(e)
        if len(arguments) == 0:
            self._argument_space = None
        elif len(arguments) == 1:
            self._argument_space = arguments[0].ufl_function_space()._cpp_object
        else:
            raise RuntimeError("Expressions with more that one Argument not allowed.")

        def _create_expression(dtype):
            if np.issubdtype(dtype, np.float32):
                return _cpp.fem.create_expression_float32
            elif np.issubdtype(dtype, np.float64):
                return _cpp.fem.create_expression_float64
            elif np.issubdtype(dtype, np.complex64):
                return _cpp.fem.create_expression_complex64
            elif np.issubdtype(dtype, np.complex128):
                return _cpp.fem.create_expression_complex128
            else:
                raise NotImplementedError(f"Type {dtype} not supported.")

        ffi = module.ffi
        self._cpp_object = _create_expression(dtype)(
            ffi.cast("uintptr_t", ffi.addressof(self._ufcx_expression)),
            coeffs,
            constants,
            self.argument_space,
        )

    def eval(
        self,
        mesh: Mesh,
        entities: np.ndarray,
        values: typing.Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Evaluate Expression on entities.

        Args:
            mesh: Mesh to evaluate Expression on.
            entities: Entities to evaluate the Expression over. For
                cells, it is a list of cell indices. For facets, it is a
                2D array of (cell index, local facet index).
            values: Array to fill with evaluated values. If ``None``,
                storage will be allocated. Otherwise it must have shape
                ``(entities.shape[0], num_points, *value_shape)`` if
                there is not argument function and shape
                ``(entities.shape[0], num_points, *value_shape,
                argument_space_dim)`` if the Expression does have an
                argument function.

        Returns:
            Expression evaluated at points for ``entities``. Shape is
            ``(entities.shape[0], num_points, *value_shape)`` if there
            is no argument function, or shape is ``(entities.shape[0],
            num_points, *value_shape, argument_space_dim)`` if the
            Expression does have an argument function.
        """
        _entities = np.asarray(entities, dtype=np.int32)
        if (tdim := mesh.topology.dim) != (expr_dim := self._cpp_object.X().shape[1]):
            assert expr_dim == tdim - 1
            assert entities.ndim == 2, (
                "entities list should have two dimensions for expression evaluation on facets."
            )

        if self.argument_space is None:
            values_shape = (_entities.shape[0], self.X().shape[0], *self.value_shape)
        else:
            values_shape = (
                _entities.shape[0],
                self.X().shape[0],
                *self.value_shape,
                self.argument_space.element.space_dimension,
            )

        # Allocate memory for result if u was not provided
        if values is None:
            values = np.zeros(values_shape, dtype=self.dtype)
        else:
            if values.shape != values_shape:
                raise TypeError("Passed values array does not have correct shape.")
            if values.dtype != self.dtype:
                raise TypeError("Passed values array does not have correct dtype.")

        constants = _cpp.fem.pack_constants(self._cpp_object)
        coeffs = _cpp.fem.pack_coefficients(self._cpp_object, _entities)
        _cpp.fem.tabulate_expression(
            values, self._cpp_object, constants, coeffs, mesh._cpp_object, _entities
        )
        return values

    def X(self) -> np.ndarray:
        """Evaluation points on the reference cell"""
        return self._cpp_object.X()

    @property
    def ufl_expression(self):
        """Original UFL Expression."""
        return self._ufl_expression

    @property
    def value_shape(self) -> np.ndarray:
        """Value shape of the Expression."""
        return self._cpp_object.value_shape

    @property
    def value_size(self) -> int:
        """Value size of the expression."""
        return self._cpp_object.value_size

    @property
    def argument_space(self) -> typing.Optional[FunctionSpace]:
        """Argument function space if Expression has argument."""
        return self._argument_space

    @property
    def ufcx_expression(self):
        """The compiled ufcx_expression object."""
        return self._ufcx_expression

    @property
    def code(self) -> str:
        """C code strings."""
        return self._code

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self._cpp_object.dtype)


class Function(ufl.Coefficient):
    """A finite element function that is represented by a function space
    (domain, element and dofmap) and a vector holding the
    degrees-of-freedom."""

    _cpp_object: typing.Union[
        _cpp.fem.Function_complex64,
        _cpp.fem.Function_complex128,
        _cpp.fem.Function_float32,
        _cpp.fem.Function_float64,
    ]

    def __init__(
        self,
        V: FunctionSpace,
        x: typing.Optional[la.Vector] = None,
        name: typing.Optional[str] = None,
        dtype: typing.Optional[npt.DTypeLike] = None,
    ):
        """Initialize a finite element Function.

        Args:
            V: The function space that the Function is defined on.
            x: Function degree-of-freedom vector. Typically required
                only when reading a saved Function from file.
            name: Function name.
            dtype: Scalar type. Is not set, the DOLFINx default scalar
                type is used.
        """
        if x is not None:
            if dtype is None:
                dtype = x.array.dtype
            else:
                assert x.array.dtype == dtype, "Incompatible Vector and dtype."
        else:
            if dtype is None:
                dtype = default_scalar_type

        assert np.issubdtype(V.element.dtype, np.dtype(dtype).type(0).real.dtype), (
            "Incompatible FunctionSpace dtype and requested dtype."
        )

        # Create cpp Function
        def functiontype(dtype):
            if np.issubdtype(dtype, np.float32):
                return _cpp.fem.Function_float32
            elif np.issubdtype(dtype, np.float64):
                return _cpp.fem.Function_float64
            elif np.issubdtype(dtype, np.complex64):
                return _cpp.fem.Function_complex64
            elif np.issubdtype(dtype, np.complex128):
                return _cpp.fem.Function_complex128
            else:
                raise NotImplementedError(f"Type {dtype} not supported.")

        if x is not None:
            self._cpp_object = functiontype(dtype)(V._cpp_object, x._cpp_object)  # type: ignore
        else:
            self._cpp_object = functiontype(dtype)(V._cpp_object)  # type: ignore

        # Initialize the ufl.FunctionSpace
        super().__init__(V.ufl_function_space())

        # Set name
        if name is None:
            self.name = "f"
        else:
            self.name = name

        # Store DOLFINx FunctionSpace object
        self._V = V

        # Store Python wrapper around the underlying Vector
        self._x = la.Vector(self._cpp_object.x)

    @property
    def function_space(self) -> FunctionSpace:
        """The FunctionSpace that the Function is defined on."""
        return self._V

    def eval(self, x: npt.ArrayLike, cells: npt.ArrayLike, u=None) -> np.ndarray:
        """Evaluate Function at points x.

        Points where x has shape (num_points, 3), and cells has shape
        (num_points,) and cell[i] is the index of the cell containing
        point x[i]. If the cell index is negative the point is ignored.
        """

        # Make sure input coordinates are a NumPy array
        _x = np.asarray(x, dtype=self._V.mesh.geometry.x.dtype)
        assert _x.ndim < 3
        if len(_x) == 0:
            _x = np.zeros((0, 3), dtype=self._V.mesh.geometry.x.dtype)
        else:
            shape0 = _x.shape[0] if _x.ndim == 2 else 1
            _x = np.reshape(_x, (shape0, -1))
        num_points = _x.shape[0]
        if _x.shape[1] != 3:
            raise ValueError("Coordinate(s) for Function evaluation must have length 3.")

        # Make sure cells are a NumPy array
        _cells = np.asarray(cells, dtype=np.int32)
        assert _cells.ndim < 2
        num_points_c = _cells.shape[0] if _cells.ndim == 1 else 1
        _cells = np.reshape(_cells, num_points_c)

        # Allocate memory for return value if not provided
        if u is None:
            value_size = self._V.value_size
            u = np.empty((num_points, value_size), self.dtype)

        self._cpp_object.eval(_x, _cells, u)  # type: ignore
        if num_points == 1:
            u = np.reshape(u, (-1,))
        return u

    def interpolate_nonmatching(
        self, u0: Function, cells: npt.NDArray[np.int32], interpolation_data: PointOwnershipData
    ) -> None:
        """Interpolate a Function defined on one mesh to a function
        defined on a different mesh.

        Args:
            u0: The Function to interpolate.
            cells: The cells to interpolate over. If ``None`` then all
                cells are interpolated over.
            interpolation_data: Data needed to interpolate functions
                defined on other meshes. Created by
                :func:`dolfinx.fem.create_interpolation_data`.
        """
        self._cpp_object.interpolate(u0._cpp_object, cells, interpolation_data._cpp_object)  # type: ignore

    def interpolate(
        self,
        u0: typing.Union[Callable, Expression, Function],
        cells0: typing.Optional[np.ndarray] = None,
        cells1: typing.Optional[np.ndarray] = None,
    ) -> None:
        """Interpolate an expression.

        Args:
            u0: Callable function, Expression or Function to
               interpolate.
            cells0: Cells in mesh associated with ``u0`` to interpolate
                over. If ``None`` then all cells are interpolated over.
            cells1: Cells in the mesh associated with ``self`` to
                interpolate over. If ``None``, then taken to be the same
                cells as ``cells0``. If ``cells1`` is not ``None``, then
                it must have the same length as ``cells0``.
        """
        if cells0 is None:
            mesh = self.function_space.mesh
            map = mesh.topology.index_map(mesh.topology.dim)
            cells0 = np.arange(map.size_local + map.num_ghosts, dtype=np.int32)

        if cells1 is None:
            cells1 = np.arange(0, dtype=np.int32)

        @singledispatch
        def _interpolate(u0):
            """Interpolate a cpp.fem.Function."""
            self._cpp_object.interpolate(u0, cells0, cells1)  # type: ignore

        @_interpolate.register(Function)
        def _(u0: Function):
            """Interpolate a fem.Function."""
            self._cpp_object.interpolate(u0._cpp_object, cells0, cells1)  # type: ignore

        @_interpolate.register(int)
        def _(u0_ptr: int):
            """Interpolate using a pointer to a function f(x)."""
            self._cpp_object.interpolate_ptr(u0_ptr, cells0)  # type: ignore

        @_interpolate.register(Expression)
        def _(e0: Expression):
            """Interpolate a fem.Expression."""
            self._cpp_object.interpolate(e0._cpp_object, cells0, cells1)  # type: ignore

        try:
            # u is a Function or Expression (or pointer to one)
            _interpolate(u0)
        except TypeError:
            # u0 is callable
            assert callable(u0)
            x = _cpp.fem.interpolation_coords(
                self._V.element._cpp_object, self._V.mesh.geometry._cpp_object, cells0
            )
            self._cpp_object.interpolate(np.asarray(u0(x), dtype=self.dtype), cells0)  # type: ignore

    def copy(self) -> Function:
        """Create a copy of the Function.

        The function space is shared and the degree-of-freedom vector is
        copied.

        Returns:
            A new Function with a copy of the degree-of-freedom vector.
        """
        return Function(
            self.function_space, la.Vector(type(self.x._cpp_object)(self.x._cpp_object))
        )

    @property
    def x(self) -> la.Vector:
        """Vector holding the degrees-of-freedom."""
        return self._x

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self._cpp_object.x.array.dtype)

    @property
    def name(self) -> str:
        """Name of the Function."""
        return self._cpp_object.name  # type: ignore

    @name.setter
    def name(self, name):
        self._cpp_object.name = name

    def __str__(self):
        """Pretty print representation."""
        return self.name

    def sub(self, i: int) -> Function:
        """Return a sub-function (a view into the ``Function``).

        Sub-functions are indexed ``i = 0, ..., N-1``, where ``N`` is
        the number of sub-spaces.

        Args:
            i: Index of the sub-function to extract.

        Returns:
            A view into the parent ``Function``.

        Note:
            If the sub-Function is re-used, for performance reasons the
            returned ``Function`` should be stored by the caller to
            avoid repeated re-computation of the subspace.
        """
        return Function(self._V.sub(i), self.x, name=f"{self!s}_{i}")

    def split(self) -> tuple[Function, ...]:
        """Extract (any) sub-functions.

        A sub-function can be extracted from a discrete function that is
        in a mixed, vector, or tensor FunctionSpace. The sub-function
        resides in the subspace of the mixed space.

        Returns:
            First level of subspaces of the function space.
        """
        num_sub_spaces = self.function_space.num_sub_spaces
        if num_sub_spaces == 1:
            raise RuntimeError("No subfunctions to extract")
        return tuple(self.sub(i) for i in range(num_sub_spaces))

    def collapse(self) -> Function:
        u_collapsed = self._cpp_object.collapse()  # type: ignore
        V_collapsed = FunctionSpace(
            self.function_space._mesh,
            self.ufl_element(),  # type: ignore
            u_collapsed.function_space,
        )
        return Function(V_collapsed, la.Vector(u_collapsed.x))


class ElementMetaData(typing.NamedTuple):
    """Data for representing a finite element

    :param family: Element type.
    :param degree: Polynomial degree of the element.
    :param shape: Shape for vector/tensor valued elements that are
        constructed from blocked scalar elements (e.g., Lagrange).
    :param symmetry: Symmetry option for blocked tensor elements.
    """

    family: str
    degree: int
    shape: typing.Optional[tuple[int, ...]] = None
    symmetry: typing.Optional[bool] = None


def functionspace(
    mesh: Mesh,
    element: typing.Union[
        ufl.finiteelement.AbstractFiniteElement,
        ElementMetaData,
        tuple[str, int],
        tuple[str, int, tuple],
        tuple[str, int, tuple, bool],
    ],
) -> FunctionSpace:
    """Create a finite element function space.

    Args:
        mesh: Mesh that space is defined on.
        element: Finite element description.

    Returns:
        A function space.
    """
    # Create UFL element
    dtype = mesh.geometry.x.dtype
    try:
        e = ElementMetaData(*element)  # type: ignore
        ufl_e = basix.ufl.element(
            e.family,
            mesh.basix_cell(),  # type: ignore
            e.degree,
            shape=e.shape,
            symmetry=e.symmetry,
            dtype=dtype,
        )
    except TypeError:
        ufl_e = element  # type: ignore

    # Check that element and mesh cell types match
    if ((domain := mesh.ufl_domain()) is None) or ufl_e.cell != domain.ufl_cell():
        raise ValueError("Non-matching UFL cell and mesh cell shapes.")
    # Create DOLFINx objects
    element = finiteelement(mesh.topology.cell_type, ufl_e, dtype)  # type: ignore
    cpp_dofmap = _cpp.fem.create_dofmap(mesh.comm, mesh.topology._cpp_object, element._cpp_object)  # type: ignore
    assert np.issubdtype(mesh.geometry.x.dtype, element.dtype), (  # type: ignore
        "Mesh and element dtype are not compatible."
    )

    # Initialize the cpp.FunctionSpace
    try:
        cppV = _cpp.fem.FunctionSpace_float64(mesh._cpp_object, element._cpp_object, cpp_dofmap)  # type: ignore
    except TypeError:
        cppV = _cpp.fem.FunctionSpace_float32(mesh._cpp_object, element._cpp_object, cpp_dofmap)  # type: ignore

    return FunctionSpace(mesh, ufl_e, cppV)


class FunctionSpace(ufl.FunctionSpace):
    """A space on which Functions (fields) can be defined."""

    _cpp_object: typing.Union[_cpp.fem.FunctionSpace_float32, _cpp.fem.FunctionSpace_float64]
    _mesh: Mesh

    def __init__(
        self,
        mesh: Mesh,
        element: ufl.finiteelement.AbstractFiniteElement,
        cppV: typing.Union[_cpp.fem.FunctionSpace_float32, _cpp.fem.FunctionSpace_float64],
    ):
        """Create a finite element function space.

        Note:
            This initialiser is for internal use and not normally called
            in user code. Use :func:`functionspace` to create a function
            space.

        Args:
            mesh: Mesh that space is defined on.
            element: UFL finite element.
            cppV: Compiled C++ function space.
        """
        if mesh._cpp_object is not cppV.mesh:
            raise RuntimeError("Meshes do not match in function space initialisation.")
        ufl_domain = mesh.ufl_domain()
        self._cpp_object = cppV
        self._mesh = mesh
        super().__init__(ufl_domain, element)

    def clone(self) -> FunctionSpace:
        """Create a new FunctionSpace :math:`W` which shares data with this
        FunctionSpace :math:`V`, but with a different unique integer ID.

        This function is helpful for defining mixed problems and using
        blocked linear algebra. For example, a matrix block defined on
        the spaces :math:`V \\times W` where, :math:`V` and :math:`W`
        are defined on the same finite element and mesh can be
        identified as an off-diagonal block whereas the :math:`V \\times
        V` and :math:`V \\times V` matrices can be identified as
        diagonal blocks. This is relevant for the handling of boundary
        conditions.

        Returns:
            A new function space that shares data
        """
        try:
            Vcpp = _cpp.fem.FunctionSpace_float64(
                self._cpp_object.mesh, self._cpp_object.element, self._cpp_object.dofmap
            )  # type: ignore
        except TypeError:
            Vcpp = _cpp.fem.FunctionSpace_float32(
                self._cpp_object.mesh, self._cpp_object.element, self._cpp_object.dofmap
            )  # type: ignore
        return FunctionSpace(self._mesh, self.ufl_element(), Vcpp)

    @property
    def num_sub_spaces(self) -> int:
        """Number of sub spaces."""
        return self.element.num_sub_elements

    def sub(self, i: int) -> FunctionSpace:
        """Return the i-th sub space.

        Args:
            i: Index of the subspace to extract.

        Returns:
            A subspace.

        Note:
            If the subspace is re-used, for performance reasons the
            returned subspace should be stored by the caller to avoid
            repeated re-computation of the subspace.
        """
        assert self.ufl_element().num_sub_elements > i
        sub_element = self.ufl_element().sub_elements[i]
        cppV_sub = self._cpp_object.sub([i])  # type: ignore
        return FunctionSpace(self._mesh, sub_element, cppV_sub)

    def component(self):
        """Return the component relative to the parent space."""
        return self._cpp_object.component()  # type: ignore

    def contains(self, V) -> bool:
        """Check if a space is contained in, or is the same as
        (identity), this space.

        Args:
            V: The space to check to for inclusion.

        Returns:
           `` True`` if ``V`` is contained in, or is the same as, this
           space.
        """
        return self._cpp_object.contains(V._cpp_object)  # type: ignore

    def __eq__(self, other):
        """Comparison for equality."""
        return super().__eq__(other) and self._cpp_object == other._cpp_object

    def __ne__(self, other):
        """Comparison for inequality."""
        return super().__ne__(other) or self._cpp_object != other._cpp_object

    def ufl_function_space(self) -> ufl.FunctionSpace:
        """UFL function space."""
        return self

    @cached_property
    def element(self) -> FiniteElement:
        """Function space finite element."""
        return FiniteElement(self._cpp_object.element)

    @property
    def dofmap(self) -> dofmap.DofMap:
        """Degree-of-freedom map associated with the function space."""
        return dofmap.DofMap(self._cpp_object.dofmap)  # type: ignore

    @property
    def mesh(self) -> Mesh:
        """Mesh on which the function space is defined."""
        return self._mesh

    def collapse(self) -> tuple[FunctionSpace, np.ndarray]:
        """Collapse a subspace and return a new function space and a map
        from new to old dofs.

        Returns:
            A new function space and the map from new to old
            degrees-of-freedom.
        """
        cpp_space, dofs = self._cpp_object.collapse()  # type: ignore
        V = FunctionSpace(self._mesh, self.ufl_element(), cpp_space)
        return V, dofs

    def tabulate_dof_coordinates(self) -> npt.NDArray[np.float64]:
        """Tabulate the coordinates of the degrees-of-freedom in the
        function space.

        Returns:
            Coordinates of the degrees-of-freedom.

        Note:
            This method is only for elements with point evaluation
            degrees-of-freedom.
        """
        return self._cpp_object.tabulate_dof_coordinates()  # type: ignore
