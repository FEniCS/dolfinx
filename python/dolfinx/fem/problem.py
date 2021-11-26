# -*- coding: utf-8 -*-
# Copyright (C) 2020 Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import typing

import ufl
from dolfinx import fem
from petsc4py import PETSc


class LinearProblem():
    """Class for solving a linear variational problem of the form a(u,
    v) = L(v) for all v using PETSc as a linear algebra backend.

    """

    def __init__(self, a: ufl.Form, L: ufl.Form, bcs: typing.List[fem.DirichletBC] = [],
                 u: fem.Function = None, petsc_options={}, form_compiler_parameters={}, jit_parameters={}):
        """Initialize solver for a linear variational problem.

        Parameters
        ----------
        a
            A bilinear UFL form, the left hand side of the variational problem.

        L
            A linear UFL form, the right hand side of the variational problem.

        bcs
            A list of Dirichlet boundary conditions.

        u
            The solution function. It will be created if not provided.

        petsc_options
            Parameters that is passed to the linear algebra backend
            PETSc. For available choices for the 'petsc_options' kwarg,
            see the `PETSc-documentation
            <https://www.mcs.anl.gov/petsc/documentation/index.html>`.

        form_compiler_parameters
            Parameters used in FFCx compilation of this form. Run `ffcx
            --help` at the commandline to see all available options.
            Takes priority over all other parameter values, except for
            `scalar_type` which is determined by DOLFINx.

        jit_parameters
            Parameters used in CFFI JIT compilation of C code generated
            by FFCx. See `python/dolfinx/jit.py` for all available
            parameters. Takes priority over all other parameter values.

        .. code-block:: python
            problem = LinearProblem(a, L, [bc0, bc1], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        """
        self._a = fem.Form(a, form_compiler_parameters=form_compiler_parameters, jit_parameters=jit_parameters)
        self._A = fem.create_matrix(self._a)

        self._L = fem.Form(L, form_compiler_parameters=form_compiler_parameters, jit_parameters=jit_parameters)
        self._b = fem.create_vector(self._L)

        if u is None:
            # Extract function space from TrialFunction (which is at the
            # end of the argument list as it is numbered as 1, while the
            # Test function is numbered as 0)
            self.u = fem.Function(a.arguments()[-1].ufl_function_space())
        else:
            self.u = u
        self.bcs = bcs

        self._solver = PETSc.KSP().create(self.u.function_space.mesh.comm)
        self._solver.setOperators(self._A)

        # Give PETSc solver options a unique prefix
        solver_prefix = "dolfinx_solve_{}".format(id(self))
        self._solver.setOptionsPrefix(solver_prefix)

        # Set PETSc options
        opts = PETSc.Options()
        opts.prefixPush(solver_prefix)
        for k, v in petsc_options.items():
            opts[k] = v
        opts.prefixPop()
        self._solver.setFromOptions()

    def solve(self) -> fem.Function:
        """Solve the problem."""

        # Assemble lhs
        self._A.zeroEntries()
        fem.assemble_matrix(self._A, self._a, bcs=self.bcs)
        self._A.assemble()

        # Assemble rhs
        with self._b.localForm() as b_loc:
            b_loc.set(0)
        fem.assemble_vector(self._b, self._L)

        # Apply boundary conditions to the rhs
        fem.apply_lifting(self._b, [self._a], bcs=[self.bcs])
        self._b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem.set_bc(self._b, self.bcs)

        # Solve linear system and update ghost values in the solution
        self._solver.solve(self._b, self.u.vector)
        self.u.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        return self.u

    @property
    def L(self) -> fem.Form:
        """Get the compiled linear form"""
        return self._L

    @property
    def a(self) -> fem.Form:
        """Get the compiled bilinear form"""
        return self._a

    @property
    def A(self) -> PETSc.Mat:
        """Get the matrix operator"""
        return self._A

    @property
    def b(self) -> PETSc.Vec:
        """Get the RHS vector"""
        return self._b

    @property
    def solver(self) -> PETSc.KSP:
        """Get the linear solver"""
        return self._solver


class NonlinearProblem:
    """Nonlinear problem class for solving the non-linear problem
    F(u, v) = 0 for all v in V

    """

    def __init__(self, F: ufl.form.Form, u: fem.Function, bcs: typing.List[fem.DirichletBC] = [],
                 J: ufl.form.Form = None, form_compiler_parameters={}, jit_parameters={}):
        """Initialize class that sets up structures for solving the
        non-linear problem using Newton's method, dF/du(u) du = -F(u)

        Parameters
        ----------
        F
            The PDE residual F(u, v)
        u
            The unknown
        bcs
            List of Dirichlet boundary conditions
        J
            UFL representation of the Jacobian (Optional)
        form_compiler_parameters
            Parameters used in FFCx compilation of this form. Run `ffcx
            --help` at the commandline to see all available options.
            Takes priority over all other parameter values, except for
            `scalar_type` which is determined by DOLFINx.
        jit_parameters
            Parameters used in CFFI JIT compilation of C code generated
            by FFCx. See `python/dolfinx/jit.py` for all available
            parameters. Takes priority over all other parameter values.

        .. code-block:: python
            problem = LinearProblem(F, u, [bc0, bc1])
        """
        self._L = fem.form.Form(F, form_compiler_parameters=form_compiler_parameters, jit_parameters=jit_parameters)
        # Create the Jacobian matrix, dF/du
        if J is None:
            V = u.function_space
            du = ufl.TrialFunction(V)
            J = ufl.derivative(F, u, du)

        self._a = fem.form.Form(J, form_compiler_parameters=form_compiler_parameters,
                                jit_parameters=jit_parameters)
        self.bcs = bcs

    @property
    def L(self) -> fem.Form:
        """Get the compiled linear form (the residual)"""
        return self._L

    @property
    def a(self) -> fem.Form:
        """Get the compiled bilinear form (the Jacobian)"""
        return self._a

    def form(self, x: PETSc.Vec):
        """This function is called before the residual or Jacobian is
        computed. This is usually used to update ghost values.
        Parameters
        ----------
        x
            The vector containing the latest solution
        """
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    def F(self, x: PETSc.Vec, b: PETSc.Vec):
        """Assemble the residual F into the vector b.
        Parameters
        ----------
        x
            The vector containing the latest solution
        b
            Vector to assemble the residual into
        """
        # Reset the residual vector
        with b.localForm() as b_local:
            b_local.set(0.0)
        fem.assemble_vector(b, self._L)
        # Apply boundary condition
        fem.apply_lifting(b, [self._a], bcs=[self.bcs], x0=[x], scale=-1.0)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem.set_bc(b, self.bcs, x, -1.0)

    def J(self, x: PETSc.Vec, A: PETSc.Mat):
        """Assemble the Jacobian matrix.
        Parameters
        ----------
        x
            The vector containing the latest solution
        A
            The matrix to assemble the Jacobian into
        """
        A.zeroEntries()
        fem.assemble_matrix(A, self._a, self.bcs)
        A.assemble()
