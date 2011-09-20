"""System integration tests for ufl-derivative-jit-assembly chain."""

# Copyright (C) 2011 Martin S. Alnaes
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
# First added:  2011-20-09
# Last changed: 2011-20-09

import unittest
import numpy
from dolfin import *

class IntegrateDerivatives(unittest.TestCase):

    def test_diff_then_integrate(self):

        if MPI.num_processes() > 1:
            # Not attempted, check and enable!
            print "FIXME: This unit test does not work in parallel, skipping"
            return

        # Define 1D geometry
        n = 21
        mesh = UnitInterval(n)

        # Shift and scale mesh
        x0, x1 = 1.5, 3.14
        mesh.coordinates()[:] *= (x1-x0)
        mesh.coordinates()[:] += x0

        cell = mesh.ufl_cell()
        x = cell.x
        xs = 0.1+0.8*x/x1 # scaled to be within [0.1,0.9]

        # Define list of expressions to test, and configure
        # accuracies these expressions are known to pass with.
        # The reason some functions are less accurately integrated is
        # likely that the default choice of quadrature rule is not perfect
        F_list = []
        def reg(exprs, acc=10):
            for expr in exprs:
                F_list.append((expr, acc))

        # FIXME: 0*dx and 1*dx fails in the ufl-ffc-jit framework somewhere
        monomial_list = [x**q for q in range(2, 6)]
        reg(monomial_list)
        reg([2.3*p+4.5*q for p in monomial_list for q in monomial_list])
        reg([x**x])
        reg([x**(x**2)], 8)
        reg([x**(x**3)], 6)
        reg([x**(x**4)], 2)
        # Special functions:
        reg([atan(xs)], 8)
        reg([sin(x), cos(x), exp(x)], 5)
        reg([ln(xs), pow(x, 2.7), pow(2.7, x)], 3)
        reg([asin(xs), acos(xs)], 1)
        reg([tan(xs)], 7)

        # To handle tensor algebra, make an x dependent input tensor xx and square all expressions
        def reg2(exprs, acc=10):
            for expr in exprs:
                F_list.append((inner(expr,expr), acc))
        xx = as_matrix([[2*x**2, 3*x**3], [11*x**5, 7*x**4]])
        x3v = as_vector([3*x**2, 5*x**3, 7*x**4])
        cc = as_matrix([[2, 3], [4, 5]])
        reg2([xx])
        reg2([x3v])
        reg2([cross(3*x3v, as_vector([-x3v[1], x3v[0], x3v[2]]))])
        reg2([xx.T])
        reg2([tr(xx)])
        reg2([det(xx)])
        reg2([dot(xx,0.1*xx)])
        reg2([outer(xx,xx.T)])
        reg2([dev(xx)])
        reg2([sym(xx)])
        reg2([skew(xx)])
        reg2([elem_mult(7*xx, cc)])
        reg2([elem_div(7*xx, xx+cc)])
        reg2([elem_pow(1e-3*xx, 1e-3*cc)])
        reg2([elem_pow(1e-3*cc, 1e-3*xx)])
        reg2([elem_op(lambda z: sin(z)+2, 0.03*xx)], 2) # pretty inaccurate...

        # FIXME: Add tests for all UFL operators:
        # These cause discontinuities and may be harder to test in the above fashion:
        #'inv', 'cofac',
        #'eq', 'ne', 'le', 'ge', 'lt', 'gt', 'And', 'Or', 'Not',
        #'conditional', 'sign',
        #'jump', 'avg',
        #'LiftingFunction', 'LiftingOperator',

        # FIXME: Test other derivatives: (but algorithms for operator derivatives are the same!):
        #'variable', 'diff',
        #'Dx', 'grad', 'div', 'curl', 'rot', 'Dn', 'exterior_derivative',

        debug = 0
        for F, acc in F_list:
            # Apply UFL differentiation
            f = diff(F, x)
            if debug:
                print F
                print x
                print f

            # Apply integration with DOLFIN
            # (also passes through form compilation and jit)
            M = f*dx
            if debug:
                print M
                print M.compute_form_data().preprocessed_form
            f_integral = assemble(M, mesh=mesh)

            # Compute integral of f manually from anti-derivative F
            # (passes through PyDOLFIN interface and uses UFL evaluation)
            F_diff = F((x1,)) - F((x0,))

            # Compare results. Using custom relative delta instead
            # of decimal digits here because some numbers are >> 1.
            delta = min(abs(f_integral), abs(F_diff)) * 10**-acc
            self.assertAlmostEqual(f_integral, F_diff, delta=delta)

    def test_div_grad_then_integrate_over_cells_and_boundary(self):

        if MPI.num_processes() > 1:
            # Not attempted, check and enable!
            print "FIXME: This unit test does not work in parallel, skipping"
            return

        # Define 1D geometry
        n = 21
        mesh = UnitSquare(n, n)

        # Shift and scale mesh
        x0, x1 = 1.5, 3.14
        mesh.coordinates()[:] *= (x1-x0)
        mesh.coordinates()[:] += x0

        cell = mesh.ufl_cell()
        x = cell.x
        #xs = 0.1+0.8*x/x1 # scaled to be within [0.1,0.9]
        n = cell.n

        # FIXME: Test all operators in 2D as well:
        F_list = []

        for F,acc in F_list:

            # Integrate over domain and its boundary
            int_dx = assemble(div(grad(F))*dx, mesh=mesh)
            int_ds = assemble(dot(grad(F), n)*ds, mesh=mesh)

            # Compare results. Using custom relative delta instead
            # of decimal digits here because some numbers are >> 1.
            delta = min(abs(f_integral), abs(F_diff)) * 10**-acc
            self.assertAlmostEqual(f_integral, F_diff, delta=delta)


if __name__ == "__main__":
    print ""
    print "Testing DOLFIN integration of UFL derivatives"
    print "---------------------------------------------"
    unittest.main()
