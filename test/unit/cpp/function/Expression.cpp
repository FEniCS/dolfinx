// Copyright (C) 2007 Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Garth N. Wells, 2008.
// Modified by Johannes Ring, 2009.
// Modified by Benjamin Kehlet 2012
//
// First added:  2007-05-24
// Last changed: 2014-08-12
//
// Unit tests for the function library


#include <dolfin.h>
#include "Projection.h"

#include <gtest/gtest.h>

using namespace dolfin;



// Test rewritten using Google Test
TEST(Eval, testArbitraryEval) { 
    class F0 : public Expression
    {
    public:
      F0() {}
      void eval(Array<double>& values, const Array<double>& x) const
      {
        values[0] = sin(3.0*x[0])*sin(3.0*x[1])*sin(3.0*x[2]);
      }
    };

    class F1 : public Expression
    {
    public:
      F1() {}
      void eval(Array<double>& values, const Array<double>& x) const
      {
        values[0] = 1.0 + 3.0*x[0] + 4.0*x[1] + 0.5*x[2];
      }
    };

    UnitCubeMesh mesh(8, 8, 8);

    Array<double> x(3);
    x[0] = 0.31; x[1] = 0.32; x[2] = 0.33;

    Array<double> u0(1);
    Array<double> u1(1);

    // User-defined functions (one from finite element space, one not)
    F0 f0;
    F1 f1;

    // Test evaluation of a user-defined function
    f0.eval(u0, x);
    ASSERT_NEAR(u0[0],
      sin(3.0*x[0])*sin(3.0*x[1])*sin(3.0*x[2]),
            DOLFIN_EPS);

    // Test for single core only
    if (dolfin::MPI::size(mesh.mpi_comm()) == 1)
    {
      // Test evaluation of a discrete function
      Projection::FunctionSpace V(mesh);
      Projection::BilinearForm a(V, V);
      Projection::LinearForm L(V);
      L.f = f1;
      Function g(V);
      solve(a == L, g);

      const double tol = 1.0e-6;
      f1.eval(u0, x);
      g.eval(u1, x);
      ASSERT_NEAR(u0[0], u1[0], tol);
    }

}

// Test all
int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}