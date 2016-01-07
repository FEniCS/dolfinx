// Copyright (C) 2011 Garth N. Wells
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
// First added:  2011-01-05
// Last changed: 2012-07-05
//
// This demo solves the Stokes equations using an iterative linear solver.
// Its also demonstrates how to use a precontioner matrix that is different
// from the matrix being solved.
//
// Note that the sign for the pressure has been flipped for symmetry.
//

#include <dolfin.h>
#include "Stokes.h"
#include "StokesPreconditioner.h"

using namespace dolfin;

#if defined(HAS_PETSC) || defined(HAS_TRILINOS)

int main()
{
  // Sub domain for left-hand side
  class Left : public SubDomain
  {
    bool inside(const Array<double>& x, bool on_boundary) const
    {
      return x[0] < DOLFIN_EPS;
    }
  };

  // Sub domain for right-hand side
  class Right : public SubDomain
  {
    bool inside(const Array<double>& x, bool on_boundary) const
    {
      return std::abs(1.0 - x[0]) < DOLFIN_EPS;
    }
  };

  // Sub domain for top and bottom
  class TopBottom : public SubDomain
  {
    bool inside(const Array<double>& x, bool on_boundary) const
    {
      return std::abs(1.0 - x[1]) < DOLFIN_EPS || std::abs(x[1]) < DOLFIN_EPS;
    }
  };

  // Function for inflow boundary condition for velocity
  class Inflow : public Expression
  {
  public:

    Inflow() : Expression(3) {}

    void eval(Array<double>& values, const Array<double>& x) const
    {
      values[0] = -sin(x[1]*DOLFIN_PI);
      values[1] = 0.0;
      values[2] = 0.0;
    }

  };

  // Check for available preconditioners
  if (!has_krylov_solver_preconditioner("amg"))
  {
    info("Sorry, this demo is only available when DOLFIN is compiled with AMG "
	 "preconditioner, Hypre or ML.");
    return 0;
  }

  // Check for available Krylov solvers
  std::string krylov_method;
  if (has_krylov_solver_method("minres"))
    krylov_method = "minres";
  else if (has_krylov_solver_method("tfqmr"))
    krylov_method = "tfqmr";
  else
  {
    info("Default linear algebra backend was not compiled with MINRES or TFQMR "
         "Krylov subspace method. Terminating.");
    return 0;
  }

  // Create mesh
  UnitCubeMesh mesh(16, 16, 16);

  // Create function space and subspaces
  Stokes::FunctionSpace W(mesh);
  auto W0 = std::make_shared<SubSpace>(W, 0);
  SubSpace W1(W, 1);

  // Set-up infow boundary condition
  auto inflow_prfofile = std::make_shared<Inflow>();
  auto right = std::make_shared<Right>();
  auto inflow = std::make_shared<DirichletBC>(W0, inflow_prfofile, right);

  // Set-up no-slip boundary condition
  auto zero_vector = std::make_shared<Constant>(0.0, 0.0, 0.0);
  auto top_bottom = std::make_shared<TopBottom>();
  auto noslip = std::make_shared<DirichletBC>(W0, zero_vector, top_bottom);

  // Create forms for the Stokes problem
  Constant f(0.0, 0.0, 0.0);
  Stokes::BilinearForm a(W, W);
  Stokes::LinearForm L(W);
  L.f = f;

  // Create solution function
  Function w(W);

  // Create form for the Stokes preconditioner
  StokesPreconditioner::BilinearForm a_P(W, W);

  // Assemble precondtioner system (P, b_dummy)
  auto P = std::make_shared<Matrix>();
  Vector b;
  assemble_system(*P, b, a_P, L, {inflow, noslip});

  // Assemble Stokes system (A, b)
  auto A = std::make_shared<Matrix>();
  assemble_system(*A, b, a, L, {inflow, noslip});

  // Create Krylov solver with specified method and preconditioner
  KrylovSolver solver(krylov_method, "amg");

  // Set operator (A) and precondtioner matrix (P)
  solver.set_operators(A, P);

  // Solve system
  solver.solve(*w.vector(), b);
  cout << "Soln norm: " << w.vector()->norm("l2") << endl;

  // Split solution
  Function u = w[0];
  Function p = w[1];

  // Save solution in VTK format
  File ufile_pvd("velocity.pvd");
  ufile_pvd << u;
  File pfile_pvd("pressure.pvd");
  pfile_pvd << p;

  // Plot solution
  plot(u);
  plot(p);
  interactive();
}

#else

int main()
{
  info("DOLFIN has not been configured with Trilinos or PETSc. Exiting.");
  return 0;
}

#endif
