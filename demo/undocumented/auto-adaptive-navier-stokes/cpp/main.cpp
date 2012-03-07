// Copyright (C) 2010 Marie E. Rognes
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
// Modified by Anders Logg 2011
//
// First added:  2010-08-19
// Last changed: 2011-10-04

#include <dolfin.h>
#include "AdaptiveNavierStokes.h"

using namespace dolfin;

// No-slip boundary
class Noslip : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return ((x[1] < DOLFIN_EPS || x[1] > 1.0 - DOLFIN_EPS) ||
            (on_boundary && abs(x[0] - 1.5) < 0.1 + DOLFIN_EPS));
  }
};

// Subdomain on which goal function should be defined
class Outflow : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return x[0] > 4.0 - DOLFIN_EPS;
  }
};

// Applied pressure (Neumann boundary condition)
class Pressure : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = (4.0 - x[0])/4.0;
  }
};

int main() {

  parameters["allow_extrapolation"] = true;

  // Create mesh and function space
  Mesh mesh("channel_with_flap.xml");
  AdaptiveNavierStokes::BilinearForm::TrialSpace W(mesh);

  // Unknown
  Function w(W);

  // Define boundary condition
  Constant u0(0.0, 0.0);
  Noslip noslip;
  MeshFunction<dolfin::uint> noslip_markers(mesh, mesh.topology().dim() - 1, 1);
  noslip.mark(noslip_markers, 0);
  SubSpace W0(W, 0);
  DirichletBC bc(W0, u0, noslip_markers, 0);

  // Create variational formulation and assign coefficients
  Constant nu(0.02);
  AdaptiveNavierStokes::LinearForm F(W);
  Pressure p0;
  F.p0 = p0;
  F.nu = nu;
  F.w = w;

  // Initialize Jacobian dF
  AdaptiveNavierStokes::BilinearForm dF(W, W);
  dF.nu = nu;
  dF.w = w;

  // Define goal functional
  AdaptiveNavierStokes::GoalFunctional M(mesh);
  M.w = w;
  Outflow outflow;
  MeshFunction<dolfin::uint> outflow_markers(mesh, mesh.topology().dim()-1, 1);
  outflow.mark(outflow_markers, 0);
  M.ds = outflow_markers;

  // Define error tolerance
  double tol = 1.e-5;

  // If no more control is wanted, do:
  // solve(F == 0, w, bc, dF, tol, M);
  // return 0;

  // Define variational problem from the variational form F, specify
  // the unknown Function w and the boundary condition bc
  NonlinearVariationalProblem pde(F, w, bc, dF);

  // Define solver
  AdaptiveNonlinearVariationalSolver solver(pde);

  // Set (precomputed) reference in adaptive solver to evaluate
  // quality of error estimates and adaptive refinement
  solver.parameters["reference"] = 0.40863917;

  // Solve problem with goal-oriented error control to given tolerance
  solver.solve(tol, M);

  // Plot solutions
  Function solution = w.leaf_node();
  plot(solution[0], "Velocity on finest mesh");
  plot(solution[1], "Pressure on finest mesh");

  // Show timings
  list_timings();

  return 0;
}
