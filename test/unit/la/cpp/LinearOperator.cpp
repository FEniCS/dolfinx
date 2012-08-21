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
    class MyMatrix : public LinearOperator
    {
    public:

      MyMatrix() :
        mesh(new UnitSquare(8, 8)),
        V(new ReactionDiffusionAction::FunctionSpace(mesh)),
        a_action(new ReactionDiffusionAction::LinearForm(V)),
        u(new Function(V)),
        LinearOperator(V->dim(), V->dim())
      {
        a_action->set_coefficient("u", u);
      }

      void mult(const GenericVector& x, GenericVector& y) const
      {
        *u->vector() = x;
        assemble(y, *a_action);
      }

    private:

      boost::shared_ptr<Mesh> mesh;
      boost::shared_ptr<FunctionSpace> V;
      boost::shared_ptr<Form> a_action;
      boost::shared_ptr<Function> u;

    };

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
