// Copyright (C) 2014 Anders Logg
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
// First added:  2014-06-10
// Last changed: 2014-06-11
//
// This demo program solves the Stokes equations on a domain defined
// by three overlapping and non-matching meshes.

#include <dolfin.h>
#include "MultiMeshStokes.h"

using namespace dolfin;

// Sub domain for Dirichlet boundary condition
class DirichletBoundary : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return on_boundary;
  }
};

// Value for inflow boundary condition for velocity
class InflowValue : public Expression
{
public:

  InflowValue() : Expression(2) {}

  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = sin(x[1]*DOLFIN_PI);
    values[1] = 0.0;
  }

};

// Subdomain for no-slip boundary
class NoslipBoundary : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return on_boundary && (near(x[1], 0.0) || near(x[1], 1.0));
  }
};

// Subdomain for inflow boundary
class InflowBoundary : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return on_boundary && near(x[0], 0.0);
  }
};

// Subdomain for outflow boundary
class OutflowBoundary : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return on_boundary && near(x[0], 1.0);
  }
};

// FIXME: Remove after testing
#include "reference.h"

int main(int argc, char* argv[])
{
  if (dolfin::MPI::size(MPI_COMM_WORLD) > 1)
  {
    info("Sorry, this demo does not (yet) run in parallel.");
    return 0;
  }

  // FIXME: Remove after testing
  // Run reference case if requested
  if (argc > 1 && strcasecmp(argv[1], "--reference") == 0)
  {
    run_reference();
    return 0;
  }

  // Increase log level
  set_log_level(DBG);

  // Don't reorder dofs (simplifies debugging)
  parameters["reorder_dofs_serial"] = false;

  // Create meshes
  int N = 16;
  UnitSquareMesh mesh_0(N, N);
  RectangleMesh  mesh_1(0.2, 0.2, 0.6, 0.6, N, N);
  RectangleMesh  mesh_2(0.4, 0.4, 0.8, 0.8, N, N);

  // Rotate overlapping mesh
  //rectangle_1.rotate(45);

  // Create function spaces
  MultiMeshStokes::FunctionSpace W0(mesh_0);
  MultiMeshStokes::FunctionSpace W1(mesh_1);
  MultiMeshStokes::FunctionSpace W2(mesh_2);

  // FIXME: Some of this stuff may be wrapped or automated later to
  // avoid needing to explicitly call add() and build()

  // Create forms
  MultiMeshStokes::BilinearForm a0(W0, W0);
  MultiMeshStokes::BilinearForm a1(W1, W1);
  MultiMeshStokes::BilinearForm a2(W2, W2);
  MultiMeshStokes::LinearForm L0(W0);
  MultiMeshStokes::LinearForm L1(W1);
  MultiMeshStokes::LinearForm L2(W2);

  // Build multimesh function space
  MultiMeshFunctionSpace W;
  W.parameters("multimesh")["quadrature_order"] = 2;
  W.add(W0);
  W.add(W1);
  W.add(W2);
  W.build();

  // Set coefficients
  Constant f(0, 0);
  L0.f = f;
  L1.f = f;
  L2.f = f;

  // Build multimesh forms
  MultiMeshForm a(W, W);
  MultiMeshForm L(W);
  a.add(a0);
  a.add(a1);
  a.add(a2);
  L.add(L0);
  L.add(L1);
  L.add(L2);
  a.build();
  L.build();

  // Create boundary values
  Constant noslip_value(0, 0);
  InflowValue inflow_value;
  Constant outflow_value(0);

  // Create subdomains for boundary conditions
  NoslipBoundary noslip_boundary;
  InflowBoundary inflow_boundary;
  OutflowBoundary outflow_boundary;

  // Create subspaces for boundary conditions
  MultiMeshSubSpace V(W, 0);
  MultiMeshSubSpace Q(W, 1);

  // Create boundary conditions
  MultiMeshDirichletBC bc0(V, noslip_value,  noslip_boundary);
  MultiMeshDirichletBC bc1(V, inflow_value,  inflow_boundary);
  MultiMeshDirichletBC bc2(Q, outflow_value, outflow_boundary);

  // Assemble linear system
  Matrix A;
  Vector b;
  MultiMeshAssembler assembler;
  assembler.assemble(A, a);
  assembler.assemble(b, L);

  // Apply boundary conditions
  bc0.apply(A, b);
  bc1.apply(A, b);
  bc2.apply(A, b);

  // Compute solution
  MultiMeshFunction u(W);
  solve(A, *u.vector(), b);

  // Save to file
  File u0_file("u0.pvd");
  File u1_file("u1.pvd");
  File u2_file("u2.pvd");
  u0_file << *u.part(0);
  u1_file << *u.part(1);
  u2_file << *u.part(2);

  // Plot solution
  plot(W.multimesh());
  plot(u.part(0), "u_0");
  plot(u.part(1), "u_1");
  plot(u.part(2), "u_2");
  interactive();

  return 0;
}
