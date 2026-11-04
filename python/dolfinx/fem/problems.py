# Copyright (C) 2026 Jack S. Hale and Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""High-level problem classes using native linear algebra objects.

Users with advanced requirements should use
:mod:`dolfinx.fem.petsc`.
"""

import typing
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

import ufl
from dolfinx import default_scalar_type
from dolfinx.fem import (
    DirichletBC,
    Form,
    Function,
    FunctionSpace,
    bcs_by_block,
    create_matrix,
    create_vector,
    extract_function_spaces,
    form,
    pack_coefficients,
    pack_constants,
)
from dolfinx.fem.assemble import (
    _apply_lifting_markers,
    _assemble_matrix_csr_markers,
    _assemble_vector_array,
    _bc_dof_markers_by_space,
    _bc_lifting_values,
)
from dolfinx.fem.forms import derivative_block
from dolfinx.la import InsertMode, MatrixCSR, Vector, norm
from dolfinx.la.superlu_dist import SuperLUDistSolver, superlu_dist_matrix, superlu_dist_solver
from dolfinx.mesh import EntityMap as EntityMap

__all__ = ["LinearProblem", "NonlinearProblem"]

_U = typing.TypeVar("_U", bound=Function | Sequence[Function])

# Dof markers on the row and column spaces, and the conditions on each
# row and column space
_BCCache = tuple[
    list[npt.NDArray[np.int8]],
    list[npt.NDArray[np.int8]],
    list[list[DirichletBC]],
    list[list[DirichletBC]],
]


# Constants and coefficients of each block of a bilinear form array
_Packed = tuple[
    list[list[npt.NDArray]],
    list[list[dict[tuple[typing.Any, int], npt.NDArray]]],
]


def _square_blocks(a: typing.Any, name: str) -> list[list[typing.Any]]:
    """Rows of a square, non-empty array of forms, as lists."""
    blocks = [list(row) for row in a]
    n = len(blocks)
    if n == 0 or any(len(row) != n for row in blocks):
        raise ValueError(f"{name} must be a square, non-empty array of forms.")
    return blocks


def _function_list(u: Function | Sequence[Function] | None) -> list[Function]:
    """Solution functions as a list, empty if ``u`` is None."""
    if u is None:
        return []
    return [u] if isinstance(u, Function) else list(u)


def _block_spaces(
    a: list[list[Form | None]], name: str
) -> tuple[list[FunctionSpace], list[FunctionSpace]]:
    """Test spaces of the block rows and trial spaces of the block columns.

    Each block must have its own function space, e.g. two blocks on one
    space ``V``, as from ``MixedFunctionSpace(V, V)``, are rejected.
    """
    V0 = extract_function_spaces(a, 0)
    V1 = extract_function_spaces(a, 1)
    for kind, spaces in (("test", V0), ("trial", V1)):
        for j, Vj in enumerate(spaces):
            if Vj is None:
                raise ValueError(
                    f"Block {'row' if kind == 'test' else 'column'} {j} of {name} has no form."
                )
            for k, Vk in enumerate(spaces[:j]):
                if Vj == Vk:
                    raise ValueError(
                        f"Blocks {k} and {j} have the same {kind} space. Each block must have "
                        "its own function space."
                    )
    return typing.cast(list[FunctionSpace], V0), typing.cast(list[FunctionSpace], V1)


def _check_function_dtypes(u: list[Function], n: int, dtype: npt.DTypeLike) -> None:
    """Check that there are ``n`` functions ``u`` of the given dtype.

    Checked before compiling forms, which takes the dtype from ``u``.
    """
    if len(u) != n:
        raise ValueError(f"u has {len(u)} functions but the problem has {n} blocks.")
    for j, uj in enumerate(u):
        if np.dtype(dtype) != uj.dtype:
            raise ValueError(f"dtype ({dtype}) does not match u[{j}].dtype ({uj.dtype}).")


def _check_function_spaces(u: list[Function], V1: list[FunctionSpace]) -> None:
    """Check that ``u[j]`` is on the trial space ``V1[j]``."""
    for j, (uj, V1j) in enumerate(zip(u, V1, strict=True)):
        if uj.function_space != V1j:
            raise ValueError(f"u[{j}] is not on the trial space of block column {j}.")


def _pack(a: list[list[Form | None]], dtype: npt.DTypeLike) -> _Packed:
    """Constants and coefficients of the blocks of ``a``, packed once.

    Empty for ``None`` blocks.
    """
    constants = [
        [np.empty(0, dtype=dtype) if aij is None else pack_constants(aij) for aij in row]
        for row in a
    ]
    coeffs = [[{} if aij is None else pack_coefficients(aij) for aij in row] for row in a]
    return constants, coeffs


def _bc_cache(
    a: list[list[Form | None]],
    V0: list[FunctionSpace],
    V1: list[FunctionSpace],
    bcs: tuple[DirichletBC, ...],
    name: str,
) -> _BCCache:
    """Dof markers and conditions by block, for conditions ``bcs``.

    Constrained rows need a diagonal entry, which assembly sets only in
    a diagonal block whose test and trial spaces are the same. Markers
    are empty on every rank when no condition applies to a space, so all
    ranks agree on whether to raise.
    """
    markers0 = _bc_dof_markers_by_space(V0, bcs)
    markers1 = _bc_dof_markers_by_space(V1, bcs)
    for i, row in enumerate(a):
        if markers0[i].size == 0 and markers1[i].size == 0:
            continue
        aii = row[i]
        if aii is None:
            raise ValueError(
                f"Boundary conditions constrain block {i}, but {name}[{i}][{i}] is None."
            )
        W0, W1 = aii.function_spaces
        if W0._cpp_object is not W1._cpp_object:
            raise ValueError(
                f"Boundary conditions constrain block {i}, but {name}[{i}][{i}] has "
                "different test and trial spaces."
            )
    return markers0, markers1, bcs_by_block(V0, bcs), bcs_by_block(V1, bcs)


def _assemble_matrix_blocks(
    A: list[list[MatrixCSR | None]],
    a: list[list[Form | None]],
    cache: _BCCache,
    packed: _Packed | None = None,
) -> None:
    """Assemble the blocks of ``a`` into ``A``; zero constrained dofs.

    ``packed`` holds the constants and coefficients of ``a`` from
    :func:`_pack`, which are packed here if not given.
    """
    markers0, markers1, _, _ = cache
    for i, row in enumerate(A):
        for j, Aij in enumerate(row):
            if Aij is not None:
                Aij.set_value(Aij.data.dtype.type(0.0))
                _assemble_matrix_csr_markers(
                    Aij,
                    a[i][j],  # type: ignore[arg-type]
                    markers0[i],
                    markers1[j],
                    constants=None if packed is None else packed[0][i][j],
                    coeffs=None if packed is None else packed[1][i][j],
                )
                Aij.scatter_reverse()


def _assemble_vector_blocks(
    b: list[Vector],
    L: list[Form | None],
    a: list[list[Form | None]],
    V1: list[FunctionSpace],
    cache: _BCCache,
    x0: list[Function] | None = None,
    alpha: float = 1.0,
    packed: _Packed | None = None,
) -> None:
    """Assemble the blocks of ``L`` into ``b`` and apply the conditions.

    Lifting by ``a`` and the constrained entries follow
    :func:`dolfinx.fem.apply_lifting` and
    :meth:`dolfinx.fem.DirichletBC.set` with ``x0`` and ``alpha``.
    ``packed`` holds the constants and coefficients of ``a`` from
    :func:`_pack`, which are packed here if not given.
    """
    _, markers1, bcs0, bcs1 = cache
    bc_values1 = _bc_lifting_values(V1, bcs1, b[0].array.dtype)
    _x0 = [] if x0 is None else [uj.x.array for uj in x0]
    lift = any(len(bcs) > 0 for bcs in bcs1)
    for i, (bi, Li) in enumerate(zip(b, L, strict=True)):
        bi.array[:] = 0.0
        if Li is not None:
            _assemble_vector_array(bi.array, Li)
        if lift:
            _apply_lifting_markers(
                bi.array,
                a[i],
                markers1,
                bc_values1,
                _x0,
                alpha,
                constants=None if packed is None else packed[0][i],
                coeffs=None if packed is None else packed[1][i],
            )
        bi.scatter_reverse(InsertMode.add)
        for bc in bcs0[i]:
            bc.set(bi.array, None if x0 is None else _x0[i], alpha)


def _factorise(
    A: list[list[MatrixCSR | None]],
    blocked: bool,
    options: dict | None,
    solver: SuperLUDistSolver | None = None,
) -> SuperLUDistSolver:
    """Copy ``A`` to SuperLU_DIST for a new or an existing ``solver``.

    An existing solver reuses the sparsity pattern of its previous
    matrix, or refactorises fully where SuperLU_DIST cannot reuse it
    (see :meth:`SuperLUDistSolver.set_A`).
    """
    # SuperLU_DIST solves in-place, so a deep copy of A is required.
    A_superlu_dist = superlu_dist_matrix(A if blocked else A[0][0])  # type: ignore[arg-type]
    if solver is None:
        solver = superlu_dist_solver(A_superlu_dist)
        for option, value in (options or {}).items():
            solver.set_option(option, value)
    else:
        solver.set_A(A_superlu_dist, "SamePattern")
    return solver


def _solve(solver: SuperLUDistSolver, b: list[Vector], x: list[Vector], blocked: bool) -> None:
    """Solve with SuperLU_DIST and update the ghost values of ``x``."""
    error = solver.solve(b, x) if blocked else solver.solve(b[0], x[0])
    if error != 0:
        raise RuntimeError(f"SuperLU_DIST returned non-zero error code: {error}")
    for xi in x:
        xi.scatter_forward()


class LinearProblem(typing.Generic[_U]):
    r"""High-level class for solving linear problems using SuperLU_DIST.

    Solves problems of the form
    :math:`\sum_j a_{ij}(u_j, v_i) = L_i(v_i) \; \forall v_i \in V_i`,
    where a single form is the case of one block, using
    :class:`dolfinx.la.superlu_dist.SuperLUDistSolver` as the linear
    solver.

    Note:
        DOLFINx must be built with SuperLU_DIST to use this class.
    """

    @typing.overload
    def __init__(
        self: "LinearProblem[Function]",
        a: ufl.Form,
        L: ufl.Form,
        bcs: Sequence[DirichletBC] | None = None,
        u: Function | None = None,
        dtype: npt.DTypeLike = default_scalar_type,
        superlu_dist_options: dict | None = None,
        form_compiler_options: dict | None = None,
        jit_options: dict | None = None,
        entity_maps: Sequence[EntityMap] | None = None,
    ) -> None: ...
    @typing.overload
    def __init__(
        self: "LinearProblem[Sequence[Function]]",
        a: Sequence[Sequence[ufl.Form | None]],
        L: Sequence[ufl.Form | None],
        bcs: Sequence[DirichletBC] | None = None,
        u: Sequence[Function] | None = None,
        dtype: npt.DTypeLike = default_scalar_type,
        superlu_dist_options: dict | None = None,
        form_compiler_options: dict | None = None,
        jit_options: dict | None = None,
        entity_maps: Sequence[EntityMap] | None = None,
    ) -> None: ...
    def __init__(
        self,
        a: ufl.Form | Sequence[Sequence[ufl.Form | None]],
        L: ufl.Form | Sequence[ufl.Form | None],
        bcs: Sequence[DirichletBC] | None = None,
        u: Function | Sequence[Function] | None = None,
        dtype: npt.DTypeLike = default_scalar_type,
        superlu_dist_options: dict | None = None,
        form_compiler_options: dict | None = None,
        jit_options: dict | None = None,
        entity_maps: Sequence[EntityMap] | None = None,
    ) -> None:
        """Initialize SuperLU_DIST solver for a linear variational problem.

        Args:
            a: Bilinear UFL form, the left-hand side of the variational
               problem, or a square array of forms for a blocked
               problem, with ``None`` for a zero block. The test spaces
               of the block rows must be distinct, and likewise the
               trial spaces of the block columns.
            L: Linear UFL form, the right-hand side of the variational
               problem, or a list with one form per block row, with
               ``None`` for a zero right-hand side.
            bcs: Dirichlet boundary conditions to apply to the variational
               problem. For a blocked problem, each condition applies to
               the blocks whose space contains its space.
            u: Solution function, or one function per block column on
               its trial space for a blocked problem. Created if not
               provided.
            dtype: Scalar type for form compilation. Must match
               ``u.dtype`` if ``u`` is provided.
            superlu_dist_options: Options passed to the SuperLU_DIST
                solver. See the SuperLU_DIST User's Guide for
                available options and values.
            form_compiler_options: Options used in FFCx compilation of
                all forms. Run ``ffcx --help`` at the command line to see
                all available options.
            jit_options: Options used in CFFI JIT compilation of C code
                generated by FFCx. See :func:`dolfinx.jit.ffcx_jit` for
                all available options. Takes priority over all other
                option values.
            entity_maps: If any trial functions, test functions, or
                coefficients in the form are not defined over the same mesh
                as the integration domain, a corresponding
                :class:`EntityMap <dolfinx.mesh.EntityMap>` must be
                provided.

        Example::

            problem = LinearProblem(a, L, bcs=bc,
                superlu_dist_options={"SymmetricMode": "YES"})
            u_h = problem.solve()

        Blocked problems take an array of forms and return one function
        per block::

            problem = LinearProblem([[a00, a01], [a10, None]], [L0, L1],
                bcs=[bc0, bc1])
            u_h, p_h = problem.solve()

        Note:
            If a boundary condition constrains block ``i``, the
            diagonal block ``a[i][i]`` must be given, with the same test
            and trial space, as the constrained rows need a diagonal
            entry.
        """
        self._blocked = isinstance(a, Sequence)
        if isinstance(L, Sequence) != self._blocked:
            raise ValueError(
                "a and L must both be single forms, or an array of forms and a list of forms."
            )
        a_blocks = _square_blocks(a, "a") if self._blocked else [[a]]
        L_blocks = list(L) if self._blocked else [L]  # type: ignore[arg-type]
        if len(L_blocks) != len(a_blocks):
            raise ValueError(f"a has {len(a_blocks)} block rows but L has {len(L_blocks)} forms.")

        _u = _function_list(u)
        if len(_u) > 0:
            _check_function_dtypes(_u, len(a_blocks), dtype)
        _dtype = _u[0].dtype if len(_u) > 0 else dtype

        def _form(f):
            return form(
                f,
                _dtype,
                form_compiler_options=form_compiler_options,
                jit_options=jit_options,
                entity_maps=entity_maps,
            )

        self._a_blocks: list[list[Form | None]] = _form(a_blocks)
        self._L_blocks: list[Form | None] = _form(L_blocks)
        self._V0, self._V1 = _block_spaces(self._a_blocks, "a")
        for i, (Li, V0i) in enumerate(zip(self._L_blocks, self._V0, strict=True)):
            if Li is not None and Li.function_spaces[0] != V0i:
                raise ValueError(f"The test space of L[{i}] differs from that of block row {i}.")
        if len(_u) > 0:
            _check_function_spaces(_u, self._V1)

        self._A_blocks: list[list[MatrixCSR | None]] = [
            [None if aij is None else create_matrix(aij) for aij in row] for row in self._a_blocks
        ]
        self._b_blocks = [create_vector(V, dtype=_dtype) for V in self._V0]
        self._x_blocks = [create_vector(V, dtype=_dtype) for V in self._V1]
        self._u_blocks = _u if len(_u) > 0 else [Function(V, dtype=_dtype) for V in self._V1]

        self.bcs = bcs

        self._superlu_dist_options = superlu_dist_options

    @property
    def bcs(self) -> tuple[DirichletBC, ...]:
        """Dirichlet boundary conditions applied to the problem.

        Assigning rebuilds the cached dof markers that :meth:`solve`
        reuses. These follow from the dofs a condition constrains,
        which are fixed once it is built, and the sequence is copied,
        so the cache cannot go stale. Condition *values* are re-read on
        every solve. An invalid assignment raises and leaves the
        previous conditions in place.
        """
        return self._bcs

    @bcs.setter
    def bcs(self, bcs: Sequence[DirichletBC] | None) -> None:
        _bcs = tuple(bcs) if bcs is not None else ()
        self._bc_cache = _bc_cache(self._a_blocks, self._V0, self._V1, _bcs, "a")
        self._bcs = _bcs

    def solve(self) -> _U:
        """Solve the problem.

        This method updates the solution function(s) ``u`` stored in
        the problem instance.

        Returns:
            The solution function, or one function per block for a
            blocked problem.
        """
        _assemble_matrix_blocks(self._A_blocks, self._a_blocks, self._bc_cache)
        solver = _factorise(self._A_blocks, self._blocked, self._superlu_dist_options)
        _assemble_vector_blocks(
            self._b_blocks, self._L_blocks, self._a_blocks, self._V1, self._bc_cache
        )
        _solve(solver, self._b_blocks, self._x_blocks, self._blocked)
        for xi, ui in zip(self._x_blocks, self._u_blocks, strict=True):
            ui.x.array[:] = xi.array
        return self.u

    @property
    def L(self) -> Form | tuple[Form | None, ...]:
        """The compiled linear form(s) representing the right-hand side."""
        return tuple(self._L_blocks) if self._blocked else self._L_blocks[0]  # type: ignore[return-value]

    @property
    def a(self) -> Form | tuple[tuple[Form | None, ...], ...]:
        """The compiled bilinear form(s) of the left-hand side."""
        if self._blocked:
            return tuple(tuple(row) for row in self._a_blocks)
        return self._a_blocks[0][0]  # type: ignore[return-value]

    @property
    def A(self) -> MatrixCSR | tuple[tuple[MatrixCSR | None, ...], ...]:
        """Left-hand side matrix, or its blocks for a blocked problem."""
        if self._blocked:
            return tuple(tuple(row) for row in self._A_blocks)
        return self._A_blocks[0][0]  # type: ignore[return-value]

    @property
    def b(self) -> Vector | tuple[Vector, ...]:
        """Right-hand side vector, one per block row if blocked."""
        return tuple(self._b_blocks) if self._blocked else self._b_blocks[0]

    @property
    def x(self) -> Vector | tuple[Vector, ...]:
        """Solution vector, one per block column for a blocked problem.

        Note:
            The vector does not share memory with the solution
            function ``u``.
        """
        return tuple(self._x_blocks) if self._blocked else self._x_blocks[0]

    @property
    def u(self) -> _U:
        """Solution function, one per block column for a blocked problem.

        Note:
            The function does not share memory with the solution
            vector ``x``.
        """
        return typing.cast(_U, tuple(self._u_blocks) if self._blocked else self._u_blocks[0])


class NonlinearProblem(typing.Generic[_U]):
    r"""High-level class for nonlinear problems, solved by Newton's method.

    Solves problems of the form
    :math:`F_i(u, v_i) = 0 \; \forall v_i \in V_i`, with
    :math:`u = (u_0, \ldots, u_N)` and a single form the case of one
    block, by Newton's method. Each Newton step solves the linear system
    of the Jacobian with
    :class:`dolfinx.la.superlu_dist.SuperLUDistSolver`.

    Note:
        DOLFINx must be built with SuperLU_DIST to use this class.
    """

    @typing.overload
    def __init__(
        self: "NonlinearProblem[Function]",
        F: ufl.Form,
        u: Function,
        bcs: Sequence[DirichletBC] | None = None,
        J: ufl.Form | None = None,
        newton_options: dict | None = None,
        superlu_dist_options: dict | None = None,
        form_compiler_options: dict | None = None,
        jit_options: dict | None = None,
        entity_maps: Sequence[EntityMap] | None = None,
    ) -> None: ...
    @typing.overload
    def __init__(
        self: "NonlinearProblem[Sequence[Function]]",
        F: Sequence[ufl.Form],
        u: Sequence[Function],
        bcs: Sequence[DirichletBC] | None = None,
        J: Sequence[Sequence[ufl.Form | None]] | None = None,
        newton_options: dict | None = None,
        superlu_dist_options: dict | None = None,
        form_compiler_options: dict | None = None,
        jit_options: dict | None = None,
        entity_maps: Sequence[EntityMap] | None = None,
    ) -> None: ...
    def __init__(
        self,
        F: ufl.Form | Sequence[ufl.Form],
        u: Function | Sequence[Function],
        bcs: Sequence[DirichletBC] | None = None,
        J: ufl.Form | Sequence[Sequence[ufl.Form | None]] | None = None,
        newton_options: dict | None = None,
        superlu_dist_options: dict | None = None,
        form_compiler_options: dict | None = None,
        jit_options: dict | None = None,
        entity_maps: Sequence[EntityMap] | None = None,
    ) -> None:
        """Initialize a Newton solver for a nonlinear variational problem.

        Args:
            F: Residual UFL form, linear in the test function, or a list
                with one residual per block.
            u: Solution function, or one function per block, holding the
                initial guess and overwritten by :meth:`solve`.
            bcs: Dirichlet boundary conditions. For a blocked problem,
                each condition applies to the blocks whose space contains
                its space.
            J: Jacobian UFL form, or a square array of forms with
                ``None`` for a zero block. If not given, it is derived
                from ``F`` with :func:`dolfinx.fem.forms.derivative_block`,
                with ``None`` for blocks that are identically zero.
            newton_options: Newton solver options:

                - ``"atol"``: absolute tolerance (default ``1e-10``),
                - ``"rtol"``: relative tolerance (default ``1e-9``),
                - ``"max_it"``: maximum number of iterations (default
                  ``50``),
                - ``"convergence"``: ``"residual"`` (default) to test the
                  norm of ``F``, or ``"incremental"`` to test the norm of
                  the Newton update,
                - ``"relaxation"``: factor applied to each update
                  (default ``1``),
                - ``"error_on_nonconvergence"``: raise if not converged
                  (default ``True``).

                The default tolerances are raised to ``100`` times the
                machine epsilon of the scalar type where that is larger.
                Given tolerances are used as they are.
            superlu_dist_options: Options passed to the SuperLU_DIST
                solver. See the SuperLU_DIST User's Guide for
                available options and values.
            form_compiler_options: Options used in FFCx compilation of
                all forms. Run ``ffcx --help`` at the command line to see
                all available options.
            jit_options: Options used in CFFI JIT compilation of C code
                generated by FFCx. See :func:`dolfinx.jit.ffcx_jit` for
                all available options. Takes priority over all other
                option values.
            entity_maps: If any trial functions, test functions, or
                coefficients in the form are not defined over the same mesh
                as the integration domain, a corresponding
                :class:`EntityMap <dolfinx.mesh.EntityMap>` must be
                provided.

        Example::

            problem = NonlinearProblem([F0, F1], [u0, u1], bcs=[bc0, bc1])
            u0, u1 = problem.solve()
            print(problem.num_iterations, problem.residual_norms)

        Note:
            Each block must have its own function space: two unknowns
            on one space ``V``, as from ``MixedFunctionSpace(V, V)``,
            are rejected.

        Note:
            If a boundary condition constrains block ``i``, the
            diagonal block ``J[i][i]`` must be given, with the same test
            and trial space, as the constrained rows need a diagonal
            entry.
        """
        self._blocked = isinstance(F, Sequence)
        if isinstance(u, Function) == self._blocked:
            raise ValueError("F and u must both be single, or both be lists.")
        F_blocks = list(F) if self._blocked else [F]  # type: ignore[arg-type]
        _u = _function_list(u)
        if len(F_blocks) != len(_u):
            raise ValueError(f"F has {len(F_blocks)} forms but u has {len(_u)} functions.")
        dtype = _u[0].dtype
        _check_function_dtypes(_u, len(_u), dtype)

        if J is None:
            # Identically zero blocks of the derived Jacobian are None
            J_ufl = derivative_block(F_blocks, _u)  # type: ignore[arg-type]
            J_blocks = [
                [None if ufl.algorithms.expand_derivatives(Jij).empty() else Jij for Jij in row]
                for row in J_ufl
            ]
        elif self._blocked:
            J_blocks = _square_blocks(J, "J")
        else:
            J_blocks = [[J]]
        if len(J_blocks) != len(F_blocks):
            raise ValueError(f"J has {len(J_blocks)} block rows but F has {len(F_blocks)} forms.")

        def _form(f):
            return form(
                f,
                dtype,
                form_compiler_options=form_compiler_options,
                jit_options=jit_options,
                entity_maps=entity_maps,
            )

        self._J_blocks: list[list[Form | None]] = _form(J_blocks)
        self._F_blocks: list[Form] = _form(F_blocks)
        self._V0, self._V1 = _block_spaces(self._J_blocks, "J")
        for i, (Fi, V0i) in enumerate(zip(self._F_blocks, self._V0, strict=True)):
            if Fi.function_spaces[0] != V0i:
                raise ValueError(f"The test space of F[{i}] differs from that of block row {i}.")
        _check_function_spaces(_u, self._V1)
        self._u_blocks = _u

        self._A_blocks: list[list[MatrixCSR | None]] = [
            [None if Jij is None else create_matrix(Jij) for Jij in row] for row in self._J_blocks
        ]
        self._b_blocks = [create_vector(V, dtype=dtype) for V in self._V0]
        self._dx_blocks = [create_vector(V, dtype=dtype) for V in self._V1]

        self.bcs = bcs

        # Newton options. Default tolerances are no tighter than the
        # scalar type can reach; given tolerances are used as they are.
        eps = 100 * float(np.finfo(_u[0].x.array.dtype).eps)
        options: dict[str, typing.Any] = {
            "atol": max(1e-10, eps),
            "rtol": max(1e-9, eps),
            "max_it": 50,
            "convergence": "residual",
            "relaxation": 1.0,
            "error_on_nonconvergence": True,
        }
        unknown = set(newton_options or {}) - set(options)
        if unknown:
            raise ValueError(f"Unknown Newton options: {sorted(unknown)}.")
        options.update(newton_options or {})
        if options["convergence"] not in ("residual", "incremental"):
            raise ValueError(
                f"Convergence criterion must be 'residual' or 'incremental', "
                f"not {options['convergence']!r}."
            )
        self._newton_options = options
        self._superlu_dist_options = superlu_dist_options

        self._solver: SuperLUDistSolver | None = None
        self._packed: _Packed | None = None
        self._num_iterations = 0
        self._residual_norms: list[float] = []
        self._converged = False

    @property
    def bcs(self) -> tuple[DirichletBC, ...]:
        """Dirichlet boundary conditions applied to the problem.

        See :attr:`LinearProblem.bcs`.
        """
        return self._bcs

    @bcs.setter
    def bcs(self, bcs: Sequence[DirichletBC] | None) -> None:
        _bcs = tuple(bcs) if bcs is not None else ()
        self._bc_cache = _bc_cache(self._J_blocks, self._V0, self._V1, _bcs, "J")
        self._bcs = _bcs

    def _residual_norm(self) -> float:
        """Assemble the residual and return its norm over all blocks.

        The residual is :math:`F(u)`, with constrained entries
        :math:`u - u_D`, so that the Newton update enforces the boundary
        conditions. The Jacobian's constants and coefficients at ``u``
        are packed for lifting if conditions apply, and reused to
        assemble the Jacobian at the same ``u``.
        """
        lift = any(len(bcs) > 0 for bcs in self._bc_cache[3])
        dtype = self._b_blocks[0].array.dtype
        self._packed = _pack(self._J_blocks, dtype) if lift else None
        _assemble_vector_blocks(
            self._b_blocks,
            self._F_blocks,  # type: ignore[arg-type]
            self._J_blocks,
            self._V1,
            self._bc_cache,
            x0=self._u_blocks,
            alpha=-1.0,
            packed=self._packed,
        )
        return float(np.sqrt(sum(norm(bi) ** 2 for bi in self._b_blocks)))

    def solve(self) -> _U:
        """Solve the problem by Newton's method.

        Starts from the current values of ``u``, which are updated in
        place. Norms are reduced over all ranks, so every rank takes
        the same convergence decision.

        Returns:
            The solution function, or one function per block for a
            blocked problem.

        Raises:
            RuntimeError: If Newton's method does not converge, or stops
                at a non-finite residual or update norm, and
                ``error_on_nonconvergence`` is set.
        """
        options = self._newton_options
        atol, rtol = options["atol"], options["rtol"]
        residual = options["convergence"] == "residual"

        self._num_iterations = 0
        r = self._residual_norm()
        self._residual_norms = [r]
        r0, dx0 = r, None
        finite = bool(np.isfinite(r))
        self._converged = finite and residual and r < atol
        while finite and not self._converged and self._num_iterations < options["max_it"]:
            # Solve J dx = F(u) and update u <- u - relaxation * dx
            if self._packed is None:
                self._packed = _pack(self._J_blocks, self._b_blocks[0].array.dtype)
            _assemble_matrix_blocks(self._A_blocks, self._J_blocks, self._bc_cache, self._packed)
            self._solver = _factorise(
                self._A_blocks, self._blocked, self._superlu_dist_options, self._solver
            )
            _solve(self._solver, self._b_blocks, self._dx_blocks, self._blocked)
            for dxj, uj in zip(self._dx_blocks, self._u_blocks, strict=True):
                uj.x.array[:] -= options["relaxation"] * dxj.array
            self._num_iterations += 1

            if not residual:
                dx = float(np.sqrt(sum(norm(dxj) ** 2 for dxj in self._dx_blocks)))
                finite = bool(np.isfinite(dx))
                dx0 = dx if dx0 is None else dx0
                self._converged = finite and (dx < atol or dx < rtol * dx0)
                if not finite or self._converged:
                    break

            r = self._residual_norm()
            self._residual_norms.append(r)
            finite = bool(np.isfinite(r))
            if residual:
                self._converged = finite and (r < atol or r < rtol * r0)

        if not self._converged and options["error_on_nonconvergence"]:
            if not finite:
                raise RuntimeError(
                    "Newton's method diverged: non-finite norm after "
                    f"{self._num_iterations} iterations."
                )
            raise RuntimeError(
                f"Newton's method did not converge in {self._num_iterations} iterations "
                f"(residual norm {self._residual_norms[-1]:.3e})."
            )
        return self.u

    @property
    def num_iterations(self) -> int:
        """Number of Newton iterations in the last :meth:`solve`."""
        return self._num_iterations

    @property
    def converged(self) -> bool:
        """Whether the last :meth:`solve` converged."""
        return self._converged

    @property
    def residual_norms(self) -> tuple[float, ...]:
        """Residual norm before each Newton iteration and after the last.

        With the ``"incremental"`` criterion, the residual after the last
        iteration is not assembled.
        """
        return tuple(self._residual_norms)

    @property
    def F(self) -> Form | tuple[Form, ...]:
        """The compiled residual form(s)."""
        return tuple(self._F_blocks) if self._blocked else self._F_blocks[0]

    @property
    def J(self) -> Form | tuple[tuple[Form | None, ...], ...]:
        """The compiled Jacobian form(s)."""
        if self._blocked:
            return tuple(tuple(row) for row in self._J_blocks)
        return self._J_blocks[0][0]  # type: ignore[return-value]

    @property
    def A(self) -> MatrixCSR | tuple[tuple[MatrixCSR | None, ...], ...]:
        """Jacobian matrix, or its blocks for a blocked problem."""
        if self._blocked:
            return tuple(tuple(row) for row in self._A_blocks)
        return self._A_blocks[0][0]  # type: ignore[return-value]

    @property
    def b(self) -> Vector | tuple[Vector, ...]:
        """Residual vector, one per block row if blocked."""
        return tuple(self._b_blocks) if self._blocked else self._b_blocks[0]

    @property
    def u(self) -> _U:
        """Solution function, one per block column if blocked."""
        return typing.cast(_U, tuple(self._u_blocks) if self._blocked else self._u_blocks[0])
