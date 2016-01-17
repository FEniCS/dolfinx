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
// Modified by Johannes Ring 2012
//
// First added:  2012-08-21
// Last changed: 2012-09-19
//
// Unit tests for matrix-free linear solvers (LinearOperator)

#include <dolfin.h>
#include "forms/ReactionDiffusion.h"
#include "forms/ReactionDiffusionAction.h"

#include <gtest/gtest.h>

using namespace dolfin;

// Backends supporting the LinearOperator interface
std::vector<std::string> backends;

TEST(TestLinearOperator, test_linear_operator) { 
    // Define linear operator
    class MyLinearOperator : public LinearOperator
    {
    public:

      MyLinearOperator(Form& a_action, Function& u)
        : LinearOperator(*u.vector(), *u.vector()),
      a_action(a_action), u(u)
      {
        // Do nothing
      }

      std::size_t size(std::size_t dim) const
      { return u.function_space()->dim(); }

      void mult(const GenericVector& x, GenericVector& y) const
      {
        // Update coefficient vector
        *u.vector() = x;

        // Assemble action
        Assembler assembler;
        assembler.assemble(y, a_action);
      }

    private:

      Form& a_action;
      Function& u;

    };

    // Iterate over backends supporting linear operators
    for (std::size_t i = 0; i < backends.size(); i++)
    {
      // Check whether backend is available
      if (!has_linear_algebra_backend(backends[i]))
  continue;

      // Skip testing Eigen in parallel
      if (dolfin::MPI::size(MPI_COMM_WORLD) > 1
          && backends[i] == "Eigen")
      {
  info("Not running Eigen test in parallel");
  continue;
      }

      // Set linear algebra backend
      parameters["linear_algebra_backend"] = backends[i];

      // Compute reference value by solving ordinary linear system
      UnitSquareMesh mesh(8, 8);
      auto V = std::make_shared<ReactionDiffusion::FunctionSpace>(mesh);
      ReactionDiffusion::BilinearForm a(V, V);
      ReactionDiffusion::LinearForm L(V);
      Constant f(1.0);
      L.f = f;
      Matrix A;
      Vector x, b;
      assemble(A, a);
      assemble(b, L);
      solve(A, x, b, "gmres", "none");
      const double norm_ref = norm(x, "l2");

      continue;

      // Solve using linear operator defined by form action
      ReactionDiffusionAction::LinearForm a_action(V);
      Function u(V);
      a_action.u = u;
      MyLinearOperator O(a_action, u);
      solve(O, x, b, "gmres", "none");
      const double norm_action = norm(x, "l2");

      // Check results
      ASSERT_NEAR(norm_ref, norm_action, 1e-10);
    }
}

// Test all
int main(int argc, char **argv) {
    // Add backends supporting the LinearOperator interface
    backends.push_back("PETSc");
    backends.push_back("Eigen");
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}




