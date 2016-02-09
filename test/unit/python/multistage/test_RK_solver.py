#!/usr/bin/env py.test

"""Unit tests for the RKSolver interface"""

# Copyright (C) 2013 Johan Hake
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

from __future__ import print_function

import pytest
from dolfin import *
import numpy as np

from dolfin_utils.test import *


def convergence_order(errors, base = 2):
    import math
    orders = [0.0] * (len(errors)-1)
    for i in range(len(errors)-1):
        try:
            orders[i] = math.log(errors[i]/errors[i+1], base)
        except ZeroDivisionError:
            orders[i] = numpy.nan

    return orders

@pytest.mark.slow
@skip_64bit_int  # The linear solver can fail wit 64-bit indices
@skip_in_parallel
def test_butcher_schemes_scalar():

    LEVEL = cpp.get_log_level()
    cpp.set_log_level(cpp.WARNING)
    mesh = UnitSquareMesh(4, 4)

    V = FunctionSpace(mesh, "R", 0)
    u = Function(V)
    v = TestFunction(V)
    form = u*v*dx

    tstop = 1.0
    u_true = Expression("exp(t)", t=tstop, degree=2)

    for Scheme in [ForwardEuler, ExplicitMidPoint, RK4,
                   BackwardEuler, CN2, ESDIRK3, ESDIRK4]:
        scheme = Scheme(form, u)
        solver = RKSolver(scheme)
        u_errors = []
        for dt in [0.05, 0.025, 0.0125]:
            u.interpolate(Constant(1.0))
            solver.step_interval(0., tstop, dt)
            u_errors.append(u_true(0.0, 0.0) - u(0.0, 0.0))

        assert scheme.order()-min(convergence_order(u_errors))<0.1

    cpp.set_log_level(LEVEL)


@pytest.mark.slow
@skip_in_parallel
def test_butcher_schemes_vector():

    LEVEL = cpp.get_log_level()
    cpp.set_log_level(cpp.WARNING)
    mesh = UnitSquareMesh(4, 4)

    V = VectorFunctionSpace(mesh, "R", 0, dim=2)
    u = Function(V)
    v = TestFunction(V)
    form = inner(as_vector((-u[1], u[0])), v)*dx

    tstop = 1.0
    u_true = Expression(("cos(t)", "sin(t)"), t=tstop, degree=2)

    for Scheme in [ForwardEuler, ExplicitMidPoint, RK4,
                   BackwardEuler, CN2, ESDIRK3, ESDIRK4]:
        scheme = Scheme(form, u)
        solver = RKSolver(scheme)
        u_errors_0 = []
        u_errors_1 = []
        for dt in [0.05, 0.025, 0.0125]:
            u.interpolate(Constant((1.0, 0.0)))
            solver.step_interval(0., tstop, dt)
            u_errors_0.append(u_true(0.0, 0.0)[0] - u(0.0, 0.0)[0])
            u_errors_1.append(u_true(0.0, 0.0)[1] - u(0.0, 0.0)[1])

        assert scheme.order()-min(convergence_order(u_errors_0))<0.1
        assert scheme.order()-min(convergence_order(u_errors_1))<0.1

    cpp.set_log_level(LEVEL)
