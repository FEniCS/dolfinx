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

      MyLinearOperator(Form& action)
        : action(action),
          LinearOperator(action.function_space(0)->dim(),
                         action.function_space(0)->dim())
      {
        // Do nothing
      }

      void mult(const GenericVector& x, GenericVector& y) const
      {
        //*u.vector() = x;
        assemble(y, action);
      }

    private:

      Form& action;

    };

    // Create form action
    UnitSquare mesh(8, 8);
    ReactionDiffusionAction::FunctionSpace V(mesh);
    ReactionDiffusionAction::LinearForm action(V);
    Function u(V);
    action.u = u;

    // Create linear operator
    MyLinearOperator A(action);

    // Solve linear system
    Vector x;
    Vector b(V.dim());
    b = 1.0;
    solve(A, x, b, "gmres");
  }

};

CPPUNIT_TEST_SUITE_REGISTRATION(TestLinearOperator);

int main()
{
  // FIXME: Testing
  set_log_level(DBG);

  DOLFIN_TEST;
}
