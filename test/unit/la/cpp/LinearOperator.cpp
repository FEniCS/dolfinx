// Copyright (C) 2012 Anders Logg
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
// First added:  2012-08-21
// Last changed: 2012-08-23
//
// Unit tests for matrix-free linear solvers (LinearOperator)

#include <dolfin.h>
#include <dolfin/common/unittest.h>
#include "forms/ReactionDiffusion.h"
#include "forms/ReactionDiffusionAction.h"

using namespace dolfin;

class TestLinearOperator : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(TestLinearOperator);
  CPPUNIT_TEST(test_linear_operator);
  CPPUNIT_TEST_SUITE_END();

public:

  void test_linear_operator()
  {
    // Define linear operator
    class MyLinearOperator : public LinearOperator
    {
    public:

      MyLinearOperator(Form& action, Function& u)
        : action(action), u(u),
          LinearOperator(u.function_space()->dim(),
                         u.function_space()->dim())
      {
        // Do nothing
      }

      void mult(const GenericVector& x, GenericVector& y) const
      {
        // Update coefficient vector
        *u.vector() = x;

        // Assemble action
        assemble(y, action, false);
      }

    private:

      Form& action;
      Function& u;

    };

    // Compute reference value by solving ordinary linear system
    UnitSquare mesh(8, 8);
    ReactionDiffusion::FunctionSpace V(mesh);
    ReactionDiffusion::BilinearForm a(V, V);
    Matrix A;
    Vector x;
    Vector b(V.dim());
    b = 1.0;
    assemble(A, a);
    solve(A, x, b, "gmres");
    const double norm_ref = norm(x, "l2");

    // Solve using linear operator defined by form action
    ReactionDiffusionAction::LinearForm action(V);
    Function u(V);
    action.u = u;
    MyLinearOperator O(action, u);
    solve(O, x, b, "gmres");
    const double norm_action = norm(x, "l2");

    // Check results
    CPPUNIT_ASSERT_DOUBLES_EQUAL(norm_ref, norm_action, 1e-10);
  }

};

CPPUNIT_TEST_SUITE_REGISTRATION(TestLinearOperator);

int main()
{
  DOLFIN_TEST;
}
