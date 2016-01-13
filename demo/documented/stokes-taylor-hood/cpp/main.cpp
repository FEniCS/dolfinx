// Copyright (C) 2006-2008 Anders Logg
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
// First added:  2006-02-09
// Last changed: 2012-07-05
//
// This demo solves the Stokes equations, using quadratic elements for
// the velocity and first degree elements for the pressure
// (Taylor-Hood elements). The sub domains for the different boundary
// conditions used in this simulation are computed by the demo program
// in src/demo/mesh/subdomains.

#include <dolfin.h>
#include "Stokes.h"

using namespace dolfin;

int main()
{
  // Function for no-slip boundary condition for velocity
  class Noslip : public Expression
  {
  public:

    Noslip() : Expression(2) {}

    void eval(Array<double>& values, const Array<double>& x) const
    {
      values[0] = 0.0;
      values[1] = 0.0;
    }

  };

  // Function for inflow boundary condition for velocity
  class Inflow : public Expression
  {
  public:

    Inflow() : Expression(2) {}

    void eval(Array<double>& values, const Array<double>& x) const
    {
      values[0] = -sin(x[1]*DOLFIN_PI);
      values[1] = 0.0;
    }

  };

  // Read mesh and sub domain markers
  Mesh mesh("../dolfin_fine.xml.gz");
  auto sub_domains = std::make_shared<MeshFunction<std::size_t>>(mesh, "../dolfin_fine_subdomains.xml.gz");

  // Create function space
  auto W = std::make_shared<Stokes::FunctionSpace>(mesh);

  // Create functions for boundary conditions
  auto noslip = std::make_shared<Noslip>();
  auto inflow = std::make_shared<Inflow>();
  auto zero = std::make_shared<Constant>(0.0);

  // No-slip boundary condition for velocity
  DirichletBC bc0(W->sub(0), noslip, sub_domains, 0);

  // Inflow boundary condition for velocity
  DirichletBC bc1(W->sub(0), inflow, sub_domains, 1);

  // Collect boundary conditions
  std::vector<const DirichletBC*> bcs = {{&bc0, &bc1}};

  // Define variational problem
  Constant f(0.0, 0.0);
  Stokes::BilinearForm a(W, W);
  Stokes::LinearForm L(W);
  L.f = f;

  // Compute solution
  Function w(W);
  solve(a == L, w, bcs);
  Function u = w[0];
  Function p = w[1];

  // Save solution in VTK format
  File ufile_pvd("velocity.pvd");
  ufile_pvd << u;
  File pfile_pvd("pressure.pvd");
  pfile_pvd << p;

  File pfile_mf("mf.pvd");
  pfile_mf << *sub_domains;

  // Plot solution
  plot(u);
  plot(p);
  interactive();

  return 0;
}
