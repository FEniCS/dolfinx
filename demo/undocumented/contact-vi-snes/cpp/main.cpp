// Copyright (C) 2012 Corrado Maurini
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
// Modified by Corrado Maurini 2013
// Modified by Johannes Ring 2013
//
// First added:  2012-09-03
// Last changed: 2013-11-21
//
// This demo program uses the PETSc nonlinear solver for variational
// inequalities to solve a contact mechanics problems in FEniCS.  The
// example considers a heavy elastic circle in a box of the same size.

#include <dolfin.h>
#include "HyperElasticity.h"

using namespace dolfin;

// Sub domain for symmetry condition
class SymmetryLine : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  { return (std::abs(x[0]) < DOLFIN_EPS); }
};

// Lower bound for displacement
class LowerBound : public Expression
{
public:
  LowerBound() : Expression(2) {}
  void eval(Array<double>& values, const Array<double>& x) const
  {
    const double xmin = -1.0 - DOLFIN_EPS;
    const double ymin = -1.0;
    values[0] = xmin - x[0];
    values[1] = ymin - x[1];
  }
};

// Upper bound for displacement
class UpperBound : public Expression
{
public:
  UpperBound() : Expression(2) {}
  void eval(Array<double>& values, const Array<double>& x) const
  {
    const double xmax = 1.0 + DOLFIN_EPS;
    const double ymax = 2.0;
    values[0] = xmax - x[0];
    values[1] = ymax - x[1];
  }
};

int main()
{
#ifdef HAS_PETSC

  // Read mesh and create function space
  auto mesh = std::make_shared<Mesh>("../circle_yplane.xml.gz");

  // Create function space
  auto V = std::make_shared<HyperElasticity::FunctionSpace>(mesh);

  // Create Dirichlet boundary conditions
  auto zero = std::make_shared<Constant>(0.0);
  auto s = std::make_shared<SymmetryLine>();
  auto bc = std::make_shared<DirichletBC>(V->sub(0), zero, s, "pointwise");

  // Define source and boundary traction functions
  auto B = std::make_shared<Constant>(0.0, -0.05);

  // Define solution function
  auto u = std::make_shared<Function>(V);

  // Set material parameters
  const double E  = 10.0;
  const double nu = 0.3;
  auto mu = std::make_shared<Constant>(E/(2*(1 + nu)));
  auto lambda = std::make_shared<Constant>(E*nu/((1.0 + nu)*(1.0 - 2.0*nu)));

  // Create (linear) form defining (nonlinear) variational problem
  auto F = std::make_shared<HyperElasticity::ResidualForm>(V);
  F->mu = mu; F->lmbda = lambda; F->B = B; F->u = u;

  // Create jacobian dF = F' (for use in nonlinear solver).
  auto J = std::make_shared<HyperElasticity::JacobianForm>(V, V);
  J->mu = mu; J->lmbda = lambda; J->u = u;

  // Interpolate expression for upper bound
  UpperBound umax_exp;
  Function umax(V);
  umax.interpolate(umax_exp);

  // Interpolate expression for lower bound
  LowerBound umin_exp;
  Function umin(V);
  umin.interpolate(umin_exp);

  // Set up the non-linear problem
  std::vector<std::shared_ptr<const DirichletBC>> bcs = {bc};
  auto problem = std::make_shared<NonlinearVariationalProblem>(F, u, bcs, J);
  problem->set_bounds(umin, umax);

  // Set up the non-linear solver
  NonlinearVariationalSolver solver(problem);
  solver.parameters["nonlinear_solver"] = "snes";
  solver.parameters("snes_solver")["linear_solver"] = "lu";
  solver.parameters("snes_solver")["maximum_iterations"] = 20;
  solver.parameters("snes_solver")["report"] = true;
  solver.parameters("snes_solver")["error_on_nonconvergence"] = false;

  // Solve the problems
  std::pair<std::size_t, bool> out;
  out = solver.solve();

  // Check for convergence. Convergence is one modifies the loading
  // and the mesh size
  cout << out.second;
  if (out.second != true)
  {
    warning("This demo is a complex nonlinear problem. Convergence is not guaranteed when modifying parameters.");
  }

  // Save solution in VTK format
  File file("displacement.pvd");
  file << *u;

  // plot the current configuration
  plot(*u, "Displacement", "displacement");

  // Make plot windows interactive
  interactive();

#endif

 return 0;
}
