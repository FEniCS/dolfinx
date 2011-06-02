// Copyright (C) 2010 Anders Logg
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
// First added:  2010-08-30
// Last changed: 2011-02-04
//
// This demo program solves the incompressible Navier-Stokes equations
// on an L-shaped domain using Chorin's splitting method.

// Begin demo

#include <dolfin.h>
#include "TentativeVelocity.h"
#include "PressureUpdate.h"
#include "VelocityUpdate.h"

using namespace dolfin;

// Define noslip domain
class NoslipDomain : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return on_boundary && x[1] < 1.0 - DOLFIN_EPS && x[0] < 1.0 - DOLFIN_EPS;
  }
};

// Define inflow domain
class InflowDomain : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return x[1] > 1.0 - DOLFIN_EPS;
  }
};

// Define inflow domain
class OutflowDomain : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return x[0] > 1.0 - DOLFIN_EPS;
  }
};

// Define pressure boundary value at inflow
class InflowPressure : public Expression
{
public:

  // Constructor
  InflowPressure() : t(0) {}

  // Evaluate pressure at inflow
  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = sin(3.0*t);
  }

  // Current time
  double t;

};

int main()
{
  // Load mesh from file
  Mesh mesh("lshape.xml.gz");

  // Refine mesh
  mesh = refine(mesh);

  // Create function spaces
  VelocityUpdate::FunctionSpace V(mesh);
  PressureUpdate::FunctionSpace Q(mesh);

  // Set parameter values
  double dt = 0.01;
  double T = 3;

  // Define values for boundary conditions
  InflowPressure p_in;
  Constant zero(0);
  Constant zero_vector(0, 0);

  // Define subdomains for boundary conditions
  NoslipDomain noslip_domain;
  InflowDomain inflow_domain;
  OutflowDomain outflow_domain;

  // Define boundary conditions
  DirichletBC noslip(V, zero_vector, noslip_domain);
  DirichletBC inflow(Q, p_in, inflow_domain);
  DirichletBC outflow(Q, zero, outflow_domain);
  std::vector<DirichletBC*> bcu;
  bcu.push_back(&noslip);
  std::vector<DirichletBC*> bcp;
  bcp.push_back(&inflow);
  bcp.push_back(&outflow);

  // Create functions
  Function u0(V);
  Function u1(V);
  Function p1(Q);

  // Create coefficients
  Constant k(dt);
  Constant f(0, 0);

  // Create forms
  TentativeVelocity::BilinearForm a1(V, V);
  TentativeVelocity::LinearForm L1(V);
  PressureUpdate::BilinearForm a2(Q, Q);
  PressureUpdate::LinearForm L2(Q);
  VelocityUpdate::BilinearForm a3(V, V);
  VelocityUpdate::LinearForm L3(V);

  // Set coefficients
  a1.k = k; L1.k = k; L1.u0 = u0; L1.f = f;
  L2.k = k; L2.u1 = u1;
  L3.k = k; L3.u1 = u1; L3.p1 = p1;

  // Assemble matrices
  Matrix A1, A2, A3;
  assemble(A1, a1);
  assemble(A2, a2);
  assemble(A3, a3);

  // Create vectors
  Vector b1, b2, b3;

  // Create files for storing solution
  File ufile("velocity.pvd");
  File pfile("pressure.pvd");

  // Time-stepping
  double t = dt;
  Progress p("Time-stepping");
  while (t < T + DOLFIN_EPS)
  {
    // Update pressure boundary condition
    p_in.t = t;

    // Compute tentative velocity step
    begin("Computing tentative velocity");
    assemble(b1, L1);
    for (dolfin::uint i = 0; i < bcu.size(); i++)
      bcu[i]->apply(A1, b1);
    solve(A1, u1.vector(), b1, "gmres", "ilu");
    end();

    // Pressure correction
    begin("Computing pressure correction");
    assemble(b2, L2);
    for (dolfin::uint i = 0; i < bcp.size(); i++)
      bcp[i]->apply(A2, b2);
    solve(A2, p1.vector(), b2, "gmres", "amg_hypre");
    end();

    // Velocity correction
    begin("Computing velocity correction");
    assemble(b3, L3);
    for (dolfin::uint i = 0; i < bcu.size(); i++)
      bcu[i]->apply(A3, b3);
    solve(A3, u1.vector(), b3, "gmres", "ilu");
    end();

    // Save to file
    ufile << u1;
    pfile << p1;

    // Move to next time step
    u0 = u1;
    p = t / T;
    t += dt;
  }

  // Plot solution
  plot(p1, "Pressure");
  plot(u1, "Velocity");

  return 0;
}
