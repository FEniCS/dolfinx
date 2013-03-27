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
#
# First added:  2013-02-20
# Last changed: 2013-02-20

import unittest
from dolfin import *

parameters.form_compiler.optimize=True
parameters.form_compiler.representation="uflacs"

import numpy as np

def convergence_order(errors, base = 2):
    import math
    orders = [0.0] * (len(errors)-1)
    for i in range(len(errors)-1):
        try:
            orders[i] = math.log(errors[i]/errors[i+1], base)
        except ZeroDivisionError:
            orders[i] = np.nan
    
    return orders

class PointIntegralSolverTest(unittest.TestCase):

    def test_butcher_schemes_scalar(self):
        
        if cpp.MPI.num_processes() > 1:
            return

        #LEVEL = cpp.get_log_level()
        #cpp.set_log_level(cpp.WARNING)
        mesh = UnitSquareMesh(10, 10)
        V = FunctionSpace(mesh, "CG", 1)
        u = Function(V)
        v = TestFunction(V)
        form = u*v*dP
        
        tstop = 1.0
        u_true = Expression("exp(t)", t=tstop) 
            
        for Scheme in [ForwardEuler, ExplicitMidPoint, RK4,
                       BackwardEuler, CN2, ESDIRK3, ESDIRK4]:
            scheme = Scheme(form, u)
            info(scheme)
            solver = PointIntegralSolver(scheme)
            solver.parameters.newton_solver.report = False
            solver.parameters.newton_solver.iterations_to_retabulate_jacobian = 5
            solver.parameters.newton_solver.maximum_iterations = 12
            u_errors = []
            for dt in [0.05, 0.025, 0.0125]:
                u.interpolate(Constant(1.0))
                solver.step_interval(0., tstop, dt)
                u_errors.append(errornorm(u_true, u))
            
            self.assertTrue(scheme.order()-min(convergence_order(u_errors))<0.1)

        #cpp.set_log_level(LEVEL)

    def test_butcher_schemes_vector(self):
        
        if cpp.MPI.num_processes() > 1:
            return

        #LEVEL = cpp.get_log_level()
        #cpp.set_log_level(cpp.WARNING)
        mesh = UnitSquareMesh(10, 10)
        V = VectorFunctionSpace(mesh, "CG", 1, dim=2)
        u = Function(V)
        v = TestFunction(V)
        form = inner(as_vector((-u[1], u[0])), v)*dP

        tstop = 1.0
        u_true = Expression(("cos(t)", "sin(t)"), t=tstop)
            
        for Scheme in [ForwardEuler, ExplicitMidPoint, RK4,
                       BackwardEuler, CN2, ESDIRK3, ESDIRK4]:
          
            scheme = Scheme(form, u)
            info(scheme)
            solver = PointIntegralSolver(scheme)
            solver.parameters.newton_solver.report = False
            solver.parameters.newton_solver.iterations_to_retabulate_jacobian = 5
            solver.parameters.newton_solver.maximum_iterations = 12
            u_errors = []
            for dt in [0.05, 0.025, 0.0125]:
                u.interpolate(Constant((1.0, 0.0)))
                solver.step_interval(0., tstop, dt)
                u_errors.append(errornorm(u_true, u))
            
            self.assertTrue(scheme.order()-min(convergence_order(u_errors))<0.1)

        #cpp.set_log_level(LEVEL)

if __name__ == "__main__":
    print ""
    print "Testing PyDOLFIN PointIntegralSolver operations"
    print "------------------------------------------------"
    unittest.main()
    cpp.set_log_level(INFO)
    list_timings()
