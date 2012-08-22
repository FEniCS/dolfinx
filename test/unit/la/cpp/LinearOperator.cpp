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
// Last changed: 2012-08-21
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

      MyLinearOperator(const Form& action, uint dim)
        : action(action),
          LinearOperator(action.function_space(0)->dim(),
                         action.function_space(0)->dim())
      {
        a_action->set_coefficient("u", u);
      }

      void mult(const GenericVector& x, GenericVector& y) const
      {
        *u->vector() = x;
        assemble(y, a_action);
      }

    private:

      const Form& action;

    };

    // Create function space and forms
    UnitSquare mesh(8, 8);
    ReactionDiffusionAction::FunctionSpace V(mesh);
    ReactionDiffusionAction::LinearForm a_action(V);
    Function u(V);

    // Create linear operator
    MyLinearOperator A(a_action);

    // Solve linear system
    MyMatrix A;
    Vector x;
    Vector b(A.size(0));
    b = 1.0;
    solve(A, x, b);
  }

};

CPPUNIT_TEST_SUITE_REGISTRATION(TestLinearOperator);

int main()
{
  DOLFIN_TEST;
}
