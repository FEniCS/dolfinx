// Copyright (C) 2014-2015 Anders Logg
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
// Last changed: 2016-03-02
//
// This demo program solves the Stokes equations on a domain defined
// by three overlapping and non-matching meshes.

#include <dolfin.h>
#include "MultiMeshStokes.h"

using namespace dolfin;
using std::make_shared;

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

// Subdomain for no-slip boundary
class NoslipBoundary : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return on_boundary && (near(x[1], 0.0) || near(x[1], 1.0));
  }
};

int main(int argc, char* argv[])
{
  if (dolfin::MPI::size(MPI_COMM_WORLD) > 1)
  {
    info("Sorry, this demo does not (yet) run in parallel.");
    return 0;
  }

  // Create meshes
  auto mesh_0 = make_shared<UnitSquareMesh>(16, 16);
  auto mesh_1 = make_shared<RectangleMesh>(Point(0.2, 0.2), Point(0.6, 0.6), 8, 8);
  auto mesh_2 = make_shared<RectangleMesh>(Point(0.4, 0.4), Point(0.8, 0.8), 8, 8);

  // Build multimesh
  auto multimesh = make_shared<MultiMesh>();
  multimesh->add(mesh_0);
  multimesh->add(mesh_1);
  multimesh->add(mesh_2);
  multimesh->build();

  // Create function space
  auto W = make_shared<MultiMeshStokes::MultiMeshFunctionSpace>(multimesh);

  // Create forms
  auto a = make_shared<MultiMeshStokes::MultiMeshBilinearForm>(W, W);
  auto L = make_shared<MultiMeshStokes::MultiMeshLinearForm>(W);

  // Attach coefficients
  auto f = make_shared<Constant>(0, 0);
  L->f = f;

  // Assemble linear system
  auto A = make_shared<Matrix>();
  auto b = make_shared<Vector>();
  assemble_multimesh(*A, *a);
  assemble_multimesh(*b, *L);

  // Create boundary values
  auto inflow_value  = make_shared<InflowValue>();
  auto outflow_value = make_shared<Constant>(0);
  auto noslip_value  = make_shared<Constant>(0, 0);

  // Create subdomains for boundary conditions
  auto inflow_boundary  = make_shared<InflowBoundary>();
  auto outflow_boundary = make_shared<OutflowBoundary>();
  auto noslip_boundary  = make_shared<NoslipBoundary>();

  // Create subspaces for boundary conditions
  auto V = make_shared<MultiMeshSubSpace>(*W, 0);
  auto Q = make_shared<MultiMeshSubSpace>(*W, 1);

  // Create boundary conditions
  auto bc0 = make_shared<MultiMeshDirichletBC>(V, noslip_value,  noslip_boundary);
  auto bc1 = make_shared<MultiMeshDirichletBC>(V, inflow_value,  inflow_boundary);
  auto bc2 = make_shared<MultiMeshDirichletBC>(Q, outflow_value, outflow_boundary);

  // Apply boundary conditions
  bc0->apply(*A, *b);
  bc1->apply(*A, *b);
  bc2->apply(*A, *b);

  // Compute solution
  MultiMeshFunction w(W);
  solve(*A, *w.vector(), *b);

  // Extract solution components
  Function& u0 = (*w.part(0))[0];
  Function& u1 = (*w.part(1))[0];
  Function& u2 = (*w.part(2))[0];
  Function& p0 = (*w.part(0))[1];
  Function& p1 = (*w.part(1))[1];
  Function& p2 = (*w.part(2))[1];

  // Save to file
  File u0_file("u0.pvd");
  File u1_file("u1.pvd");
  File u2_file("u2.pvd");
  File p0_file("p0.pvd");
  File p1_file("p1.pvd");
  File p2_file("p2.pvd");
  u0_file << u0;
  u1_file << u1;
  u2_file << u2;
  p0_file << p0;
  p1_file << p1;
  p2_file << p2;

  // Plot solution
  plot(W->multimesh());
  plot(u0, "u_0");
  plot(u1, "u_1");
  plot(u2, "u_2");
  plot(p0, "p_0");
  plot(p1, "p_1");
  plot(p2, "p_2");
  interactive();

  return 0;
}
