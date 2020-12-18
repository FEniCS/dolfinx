# -*- coding: utf-8 -*-
# Copyright (C) 2020 Jørgen S. Dokken
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
import typing

from petsc4py import PETSc
from ufl.equation import Equation
from dolfinx import fem


class LinearVariationalSolver():
    assemble_b = True

    def __init__(self, variational_problem: Equation, u: fem.Function,
                 bcs: typing.List[fem.DirichletBC] = [], form_compiler_parameters={}):
        """
        Initialize the linear variational solver. By initialization we mean creating the underlying PETSc solver,
        and the linear algebra structures (the matrix and vector), that can be reused for multiple solve calls.

        .. code-block:: python
            solver = LinearVariationalSolver(a==L, u, [bc0, bc1],
                                             form_compiler_parameters={"optimize": True})
        """
        if (len(variational_problem.lhs.arguments()) != 2):
            raise ValueError(
                "Variational problem has to consist of a bi-linear form,"
                + " defined with a `ufl.TrialFunction` and `ufl.TestFunction`.")
        # If a == 0 is inserted just assume that rhs is 0.
        if (variational_problem.rhs == 0):
            self.assemble_b = False
            self.b = u.vector.copy()
        else:
            self.L = fem.Form(variational_problem.rhs, form_compiler_parameters=form_compiler_parameters)
            self.b = fem.create_vector(self.L)

        self.solver = PETSc.KSP().create(u.function_space.mesh.mpi_comm())
        self.a = fem.Form(variational_problem.lhs, form_compiler_parameters=form_compiler_parameters)
        self.A = fem.create_matrix(self.a)
        self.solver.setOperators(self.A)

        self.bcs = bcs
        self.u = u

    def solve(self, petsc_options={}):
        """
        Solves the linear variational problem using PETSc.
        For available choices for the 'petsc_options' kwarg, see the
        `PETSc-documentation <https://www.mcs.anl.gov/petsc/documentation/index.html>`.
        """
        # Assemble lhs
        self.A.zeroEntries()
        fem.assemble_matrix(self.A, self.a, bcs=self.bcs)
        self.A.assemble()

        # Assemble rhs
        with self.b.localForm() as loc:
            loc.set(0)
        if self.assemble_b:
            fem.assemble_vector(self.b, self.L)

        # Apply boundary conditions
        fem.apply_lifting(self.b, [self.a], [self.bcs])
        self.b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem.set_bc(self.b, self.bcs)

        # Solve linear algebra problem using PETSc
        self.solver.setOptionsPrefix("dolfinx_solve_")
        opts = PETSc.Options()
        opts.prefixPush("dolfinx_solve_")
        for k, v in petsc_options.items():
            opts[k] = v
        opts.prefixPop()

        self.solver.setFromOptions()
        self.solver.solve(self.b, self.u.vector)
        self.u.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
