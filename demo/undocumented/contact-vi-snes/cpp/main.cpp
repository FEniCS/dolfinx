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
// Last changed: 2013-04-19
//
// This demo program uses of the interface to TAO solver for variational
// inequalities to solve a contact mechanics problems in FEniCS.
// The example considers a heavy elastic circle in a box of the same size.

#include <dolfin.h>
#include "HyperElasticity.h"

using namespace dolfin;

int main()
{
#ifdef HAS_PETSC
    // Sub domain for symmetry condition
    class SymmetryLine : public SubDomain
    {
        bool inside(const Array<double>& x, bool on_boundary) const
        {
        return (std::abs(x[0]) < DOLFIN_EPS);
        }
    };
    // Lower bound for displacement
    class LowerBound : public Expression
    {
    public:

      LowerBound() : Expression(2) {}

      void eval(Array<double>& values, const Array<double>& x) const
      {
        double xmin = -1.-DOLFIN_EPS;
        double ymin = -1.;
        values[0] = xmin-x[0];
        values[1] = ymin-x[1];
      }

    };

    // Upper bound for displacement
    class UpperBound : public Expression
    {
    public:

      UpperBound() : Expression(2) {}

      void eval(Array<double>& values, const Array<double>& x) const
      {
        double xmax = 1.+DOLFIN_EPS;
        double ymax = 2.;
        values[0] = xmax-x[0];
        values[1] = ymax-x[1];
      }

    };

  // Read mesh and create function space
#ifdef HAS_CGAL
  Circle circle(0, 0, 1);
  Mesh   mesh(circle,30);
#else
  UnitCircleMesh mesh(30);
#endif
  HyperElasticity::FunctionSpace V(mesh);

  // Create Dirichlet boundary conditions
  SubSpace V0(V, 0);
  Constant zero(0.0);
  SymmetryLine s;
  DirichletBC bc(V0, zero, s, "pointwise");
  std::vector<const DirichletBC*> bcs;
  bcs.push_back(&bc);

  // Define source and boundary traction functions
  Constant B(0.0, -0.05);

  // Define solution function
  Function u(V);

  // Set material parameters
  const double E  = 10.0;
  const double nu = 0.3;
  Constant mu(E/(2*(1 + nu)));
  Constant lambda(E*nu/((1 + nu)*(1 - 2*nu)));

  // Create (linear) form defining (nonlinear) variational problem
  HyperElasticity::ResidualForm F(V);
  F.mu = mu; F.lmbda = lambda; F.B = B; F.u = u;

  // Create jacobian dF = F' (for use in nonlinear solver).
  HyperElasticity::JacobianForm J(V, V);
  J.mu = mu; J.lmbda = lambda; J.u = u;

  // Interpolate expression for Upper bound
  UpperBound umax_exp;
  Function umax(V);
  umax.interpolate(umax_exp);

  // Interpolate expression for Lower bound
  LowerBound umin_exp;
  Function umin(V);
  umin.interpolate(umin_exp);

  // Set up the non-linear problem
  NonlinearVariationalProblem problem(F, u, bcs, J);

  // Set up the non-linear solver
  NonlinearVariationalSolver solver(problem);
  solver.parameters["nonlinear_solver"]="snes";
  solver.parameters["linear_solver"]="lu";
  solver.parameters("snes_solver")["maximum_iterations"]=20;
  solver.parameters("snes_solver")["report"]=true;
  solver.parameters("snes_solver")["error_on_nonconvergence"]=false;
  //info(solver.parameters,true);

  // Solve the problems
  std::pair<std::size_t, bool> out;
  out = solver.solve(umin,umax);

  // Check for convergence. Convergence is one modifies the loading and the mesh size.
  cout << out.second;
  if (out.second != true)
  {
    warning("This demo is a complex nonlinear problem. Convergence is not guaranteed when modifying some parameters or using PETSC 3.2.");
  }
  // Save solution in VTK format
  File file("displacement.pvd");
  file << u;

  // plot the current configuration
  plot(u,"Displacement", "displacement");

  // Make plot windows interactive
  interactive();
#endif

 return 0;
}
