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
// Last changed: 2017-09-28
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

  // Remove inactive dofs
  V->lock_inactive_dofs(*A, *b);

  // Compute solution
  auto w = make_shared<MultiMeshFunction>(W);
  solve(*A, *w->vector(), *b);

  // Save solution parts and components to file
  for (int part = 0; part < 3; part++)
  {
    XDMFFile ufile("output/u" + std::to_string(part) + ".xdmf");
    XDMFFile pfile("output/p" + std::to_string(part) + ".xdmf");
    ufile.write((*w->part(part))[0]);
    pfile.write((*w->part(part))[1]);
    ufile.close();
    pfile.close();
  }

  return 0;
}
