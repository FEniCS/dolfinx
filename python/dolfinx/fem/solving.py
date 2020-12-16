# -*- coding: utf-8 -*-
# Copyright (C) 2011 Anders Logg
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Simple interface for solving variational problems

A small Python layer on top of the C++ VariationalProblem/Solver classes
as well as the solve function.

"""

from petsc4py import PETSc

import ufl
from dolfinx import cpp, fem

# FIXME: The code is this file is outrageously convoluted because one
# function an do a number of unrelated operations, depending in the
# arguments passed.

# Problem classes need special handling since they involve JIT
# compilation


def solve(*args, **kwargs):
    """Solve variational problem a == L

    The following list explains the
    various ways in which the solve() function can be used.

    *Solving linear variational problems*

    A linear variational problem a(u, v) = L(v) for all v may be
    solved by calling solve(a == L, u, ...), where a is a bilinear
    form, L is a linear form, u is a Function (the solution). Optional
    arguments may be supplied to specify boundary conditions or solver
    parameters. Some examples are given below:

    .. code-block:: python

        solve(a == L, u)
        solve(a == L, u, bcs=bc)
        solve(a == L, u, bcs=[bc1, bc2])

        solve(a == L, u, bcs=bcs,
              petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
              form_compiler_parameters={"optimize": True})

    For available choices for the 'petsc_options' kwarg, see the
    `PETSc-documentation <https://www.mcs.anl.gov/petsc/documentation/index.html>`_

    """

    assert (len(args) > 0)
    assert isinstance(args[0], ufl.classes.Equation)
    return _solve_varproblem(*args, **kwargs)


def _solve_varproblem(*args, **kwargs):
    "Solve variational problem a == L or F == 0"

    # Extract arguments
    eq, u, bcs, J, tol, form_compiler_parameters, petsc_options \
        = _extract_args(*args, **kwargs)

    # Solve linear variational problem
    if isinstance(eq.lhs, ufl.Form) and isinstance(eq.rhs, ufl.Form):

        a = fem.Form(eq.lhs, form_compiler_parameters=form_compiler_parameters)
        L = fem.Form(eq.rhs, form_compiler_parameters=form_compiler_parameters)

        b = fem.assemble_vector(L._cpp_object)
        fem.apply_lifting(b, [a._cpp_object], [bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem.set_bc(b, bcs)

        A = fem.assemble_matrix(a._cpp_object, bcs)
        A.assemble()

        comm = L._cpp_object.mesh.mpi_comm()
        ksp = PETSc.KSP().create(comm)
        ksp.setOperators(A)
        ksp.setOptionsPrefix("dolfin_solve_")
        opts = PETSc.Options()
        opts.prefixPush("dolfin_solve_")
        for k, v in petsc_options.items():
            opts[k] = v
        opts.prefixPop()

        ksp.setFromOptions()
        ksp.solve(b, u.vector)

    # Solve nonlinear variational problem
    else:

        raise RuntimeError(
            "Non-linear variational problems should not be solved with the `solve`-command."
            + "Please use a Newton-solver.")


def _extract_args(*args, **kwargs):
    "Common extraction of arguments for _solve_varproblem[_adaptive]"

    # Check for use of valid kwargs
    valid_kwargs = [
        "bcs", "J", "tol", "form_compiler_parameters", "petsc_options"
    ]
    for kwarg in kwargs.keys():
        if kwarg not in valid_kwargs:
            raise RuntimeError(
                "Illegal keyword argument \'{}\'.".format(kwarg))

    # Extract equation
    if not len(args) >= 2:
        raise RuntimeError(
            "Missing arguments, expecting solve(lhs == rhs, u, bcs=bcs), where bcs is optional"
        )

    if len(args) > 3:
        raise RuntimeError(
            "Too many arguments, expecting solve(lhs == rhs, u, bcs=bcs), where bcs is optional"
        )

    # Extract equation
    eq = _extract_eq(args[0])

    # Extract solution function
    u = _extract_u(args[1])

    # Extract boundary conditions
    if len(args) > 2:
        bcs = _extract_bcs(args[2])
    elif "bcs" in kwargs:
        bcs = _extract_bcs(kwargs["bcs"])
    else:
        bcs = []

    # Extract Jacobian
    J = kwargs.get("J", None)
    if J is not None and not isinstance(J, ufl.Form):
        raise RuntimeError(
            "Solve variational problem. Expecting Jacobian J to be a UFL Form."
        )

    # Extract tolerance
    tol = kwargs.get("tol", None)
    if tol is not None and not (isinstance(tol, (float, int)) and tol >= 0.0):
        raise RuntimeError(
            "Solve variational problem. Expecting tolerance tol to be a non-negative number."
        )

    # Extract functional
    M = kwargs.get("M", None)
    if M is not None and not isinstance(M, ufl.Form):
        raise RuntimeError(
            "Solve variational problem. Expecting goal functional M to be a UFL Form."
        )

    # Extract parameters
    form_compiler_parameters = kwargs.get("form_compiler_parameters", {})
    petsc_options = kwargs.get("petsc_options", {})

    return eq, u, bcs, J, tol, M, form_compiler_parameters, petsc_options


def _extract_eq(eq):
    "Extract and check argument eq"
    if not isinstance(eq, ufl.classes.Equation):
        raise RuntimeError(
            "Solve variational problem. Expecting first argument to be an Equation."
        )

    return eq


def _extract_u(u):
    "Extract and check argument u"
    if not isinstance(u, fem.Function):
        raise RuntimeError("Expecting second argument to be a Function.")
    return u


def _extract_bcs(bcs):
    "Extract and check argument bcs"
    if bcs is None:
        bcs = []
    elif not isinstance(bcs, (list, tuple)):
        bcs = [bcs]
    for bc in bcs:
        if not isinstance(bc, cpp.fem.DirichletBC):
            raise RuntimeError(
                "solve variational problem. Unable to extract boundary condition arguments"
            )

    return bcs
