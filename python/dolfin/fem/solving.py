# -*- coding: utf-8 -*-
# Copyright (C) 2011 Anders Logg
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Simple interface for solving variational problems

A small Python layer on top of the C++ VariationalProblem/Solver classes
as well as the solve function.

"""

from petsc4py import PETSc

import ufl
from dolfin import cpp, fem, function

# FIXME: The code is this file is outrageously convoluted because one
# function an do a number of unrelated operations, depending in the
# arguments passed.

# Problem classes need special handling since they involve JIT
# compilation


# Solve function handles both linear systems and variational problems
def solve(*args, **kwargs):
    """Solve variational problem a == L or F == 0.

    The following list explains the
    various ways in which the solve() function can be used.

    *1. Solving linear variational problems*

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
              solver_parameters={"linear_solver": "lu"},
              form_compiler_parameters={"optimize": True})

    For available choices for the 'solver_parameters' kwarg, look at:

    .. code-block:: python

        info(LinearVariationalSolver.default_parameters(), True)

    *2. Solving nonlinear variational problems*

    A nonlinear variational problem F(u; v) = 0 for all v may be
    solved by calling solve(F == 0, u, ...), where the residual F is a
    linear form (linear in the test function v but possibly nonlinear
    in the unknown u) and u is a Function (the solution). Optional
    arguments may be supplied to specify boundary conditions, the
    Jacobian form or solver parameters. If the Jacobian is not
    supplied, it will be computed by automatic differentiation of the
    residual form. Some examples are given below:

    .. code-block:: python

        solve(F == 0, u)
        solve(F == 0, u, bcs=bc)
        solve(F == 0, u, bcs=[bc1, bc2])

        solve(F == 0, u, bcs, J=J,
              petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
              form_compiler_parameters={"optimize": True})


    For available choices for the 'solver_parameters' kwarg, look at:

    .. code-block:: python

        info(NonlinearVariationalSolver.default_parameters(), True)

    *4. Solving linear/nonlinear variational problems adaptively*

    Linear and nonlinear variational problems maybe solved adaptively,
    with automated goal-oriented error control. The automated error
    control algorithm is based on adaptive mesh refinement in
    combination with automated generation of dual-weighted
    residual-based error estimates and error indicators.

    An adaptive solve may be invoked by giving two additional
    arguments to the solve call, a numerical error tolerance and a
    goal functional (a Form).

    .. code-block:: python

        M = u*dx()
        tol = 1.e-6

        # Linear variational problem
        solve(a == L, u, bcs=bc, tol=tol, M=M)

        # Nonlinear problem:
        solve(F == 0, u, bcs=bc, tol=tol, M=M)

    """

    assert (len(args) > 0)
    assert isinstance(args[0], ufl.classes.Equation)
    return _solve_varproblem(*args, **kwargs)


def _solve_varproblem(*args, **kwargs):
    "Solve variational problem a == L or F == 0"

    # Extract arguments
    eq, u, bcs, J, tol, M, form_compiler_parameters, petsc_options \
        = _extract_args(*args, **kwargs)

    # Solve linear variational problem
    if isinstance(eq.lhs, ufl.Form) and isinstance(eq.rhs, ufl.Form):

        a = fem.Form(eq.lhs, form_compiler_parameters=form_compiler_parameters)
        L = fem.Form(eq.rhs, form_compiler_parameters=form_compiler_parameters)

        b = fem.assemble(L._cpp_object)
        fem.apply_lifting(b, [a._cpp_object], [bcs])
        b.ghostUpdate(
            addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem.set_bc(b, bcs)

        A = cpp.fem.create_matrix(a._cpp_object)
        A.zeroEntries()
        A = fem.assemble_matrix(a._cpp_object, bcs)
        A.assemble()

        comm = L._cpp_object.mesh().mpi_comm()
        solver = cpp.la.PETScKrylovSolver(comm)

        solver.set_options_prefix("dolfin_solve_")
        for k, v in petsc_options.items():
            cpp.la.PETScOptions.set("dolfin_solve_" + k, v)
        solver.set_from_options()

        solver.set_operator(A)
        solver.solve(u.vector(), b)

    # Solve nonlinear variational problem
    else:

        raise RuntimeError("Not implemented")
        # Create Jacobian if missing
        # if J is None:
        #    cpp.log.info(
        #        "No Jacobian form specified for nonlinear variational problem.")
        #    cpp.log.info(
        #        "Differentiating residual form F to obtain Jacobian J = F'.")
        #    F = eq.lhs
        #    J = formmanipulations.derivative(F, u)

        # Create problem
        # problem = NonlinearVariationalProblem(eq.lhs, u, bcs, J,
        #                                      form_compiler_parameters=form_compiler_parameters)

        # Create solver and call solve
        # solver = NonlinearVariationalSolver(problem)
        # solver.parameters.update(petsc_options)
        # solver.solve()


def _extract_args(*args, **kwargs):
    "Common extraction of arguments for _solve_varproblem[_adaptive]"

    # Check for use of valid kwargs
    valid_kwargs = [
        "bcs", "J", "tol", "M", "form_compiler_parameters", "petsc_options"
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
    # if hasattr(u, "cpp_object") and isinstance(u.cpp_object(), cpp.function.Function):
    #     return u.cpp_object()
    #
    # if isinstance(u, cpp.function.Function):
    #     return u
    if isinstance(u, function.Function):
        return u

    raise RuntimeError("Expecting second argument to be a Function.")
    # cpp.dolfin_error("solving.py",
    #                      "solve variational problem",
    #                      "Expecting second argument to be a Function")
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
