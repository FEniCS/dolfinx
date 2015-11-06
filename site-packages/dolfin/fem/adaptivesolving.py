# -*- coding: utf-8 -*-
"""This module provides a Python layer on top of the C++
Adaptive*VariationalSolver classes"""

# Copyright (C) 2011 Marie E. Rognes
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
#
# Modified by Anders Logg 2011
#
# First added:  2011-06-27
# Last changed: 2011-11-14

__all__ = ["AdaptiveLinearVariationalSolver",
           "AdaptiveNonlinearVariationalSolver",
           "generate_error_control",
           "generate_error_control_forms"]

import dolfin.cpp as cpp

from dolfin.fem.form import Form
from dolfin.fem.solving import LinearVariationalProblem
from dolfin.fem.solving import NonlinearVariationalProblem

from dolfin.fem.errorcontrolgenerator import DOLFINErrorControlGenerator

class AdaptiveLinearVariationalSolver(cpp.AdaptiveLinearVariationalSolver):

    # Reuse doc-string
    __doc__ = cpp.AdaptiveLinearVariationalSolver.__doc__

    def __init__(self, problem, goal):
        """
        Create AdaptiveLinearVariationalSolver

        *Arguments*

            problem (:py:class:`LinearVariationalProblem <dolfin.fem.solving.LinearVariationalProblem>`)

        """

        # Store problem
        self.problem = problem

        # Generate error control object
        ec = generate_error_control(self.problem, goal)

        # Compile goal functional separately
        p = self.problem.form_compiler_parameters
        M = Form(goal, form_compiler_parameters=p)

        # Initialize C++ base class
        cpp.AdaptiveLinearVariationalSolver.__init__(self, problem, M, ec)

    def solve(self, tol):
        """
        Solve such that the estimated error in the functional 'goal'
        is less than the given tolerance 'tol'

        *Arguments*

            tol (float)

                The error tolerance
        """

        # Call cpp.AdaptiveLinearVariationalSolver.solve directly
        cpp.AdaptiveLinearVariationalSolver.solve(self, tol)

class AdaptiveNonlinearVariationalSolver(cpp.AdaptiveNonlinearVariationalSolver):

    # Reuse doc-string
    __doc__ = cpp.AdaptiveNonlinearVariationalSolver.__doc__

    def __init__(self, problem, goal):
        """
        Create AdaptiveNonlinearVariationalSolver

        *Arguments*

            problem (:py:class:`NonlinearVariationalProblem <dolfin.fem.solving.NonlinearVariationalProblem>`)

        """

        # Store problem
        self.problem = problem

        # Generate error control object
        ec = generate_error_control(self.problem, goal)

        # Compile goal functional separately
        p = self.problem.form_compiler_parameters
        M = Form(goal, form_compiler_parameters=p)

        # Initialize C++ base class
        cpp.AdaptiveNonlinearVariationalSolver.__init__(self, problem, M, ec)

    def solve(self, tol):
        """
        Solve such that the estimated error in the functional 'goal'
        is less than the given tolerance 'tol'.

        *Arguments*

            tol (float)

                The error tolerance
        """

        # Call cpp.AdaptiveNonlinearVariationlSolver.solve with ec
        cpp.AdaptiveNonlinearVariationalSolver.solve(self, tol)

def generate_error_control(problem, goal):
    """
    Create suitable ErrorControl object from problem and the goal

    *Arguments*

        problem (:py:class:`LinearVariationalProblem <dolfin.fem.solving.LinearVariationalProblem>` or :py:class:`NonlinearVariationalProblem <dolfin.fem.solving.NonlinearVariationalProblem>`)

            The (primal) problem

        goal (:py:class:`Form <dolfin.fem.form.Form>`)

            The goal functional

    *Returns*

        :py:class:`ErrorControl <dolfin.cpp.ErrorControl>`

    """
    # Generate UFL forms to be used for error control
    (ufl_forms, is_linear) = generate_error_control_forms(problem, goal)

    # Compile generated forms
    p = problem.form_compiler_parameters
    forms = [Form(form, form_compiler_parameters=p) for form in ufl_forms]

    # Create cpp.ErrorControl object
    forms += [is_linear]  # NOTE: Lingering design inconsistency.
    ec = cpp.ErrorControl(*forms)

    # Return generated ErrorControl
    return ec

def generate_error_control_forms(problem, goal):
    """
    Create UFL forms required for initializing an ErrorControl object

    *Arguments*

        problem (:py:class:`LinearVariationalProblem <dolfin.fem.solving.LinearVariationalProblem>` or :py:class:`NonlinearVariationalProblem <dolfin.fem.solving.NonlinearVariationalProblem>`)

            The (primal) problem

        goal (:py:class:`Form <dolfin.fem.form.Form>`)

            The goal functional

    *Returns*

        (tuple of forms, bool)

    """

    msg = "Generating forms required for error control, this may take some time..."
    cpp.info(msg)

    # Paranoid checks added after introduction of multidomain features in ufl:
    for form in [goal]:
        assert len(form.ufl_domains()) > 0, "Error control got as input a form with no domain!"
        assert len(form.ufl_domains()) == 1, "Error control got as input a form with more than one domain!"

    # Extract primal forms from problem
    is_linear = True
    if isinstance(problem, LinearVariationalProblem):
        primal = (problem.a_ufl, problem.L_ufl)

        # Paranoid checks added after introduction of multidomain features in ufl:
        for form in primal:
            assert len(form.ufl_domains()) > 0, "Error control got as input a form with no domain!"
            assert len(form.ufl_domains()) == 1, "Error control got as input a form with more than one domain!"

    elif isinstance(problem, NonlinearVariationalProblem):
        is_linear = False
        primal = problem.F_ufl

        # Paranoid checks added after introduction of multidomain features in ufl:
        for form in [primal]:
            assert len(form.ufl_domains()) > 0, "Error control got as input a form with no domain!"
            assert len(form.ufl_domains()) == 1, "Error control got as input a form with more than one domain!"

    else:
        cpp.dolfin_error("adaptivesolving.py",
                         "generate forms required for error control",
                         "Unknown problem type (\"%s\")" % str(problem))

    # Extract unknown Function from problem
    u = problem.u_ufl

    # Get DOLFIN's error control generator to generate all forms
    generator = DOLFINErrorControlGenerator(primal, goal, u)
    forms = generator.generate_all_error_control_forms()

    # Paranoid checks added after introduction of multidomain features in ufl:
    for form in forms:
        assert len(form.ufl_domains()) > 0, "Error control produced a form with no domain!"
        assert len(form.ufl_domains()) == 1, "Error control produced a form with more than one domain!"

    return (forms, is_linear)
