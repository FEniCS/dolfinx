// Copyright (C) 2008 Anders Logg
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
// First added:  2008-05-23
// Last changed: 2012-07-05
//
// This demo illustrates how to set boundary conditions for meshes
// that include boundary indicators. The mesh used in this demo was
// generated with VMTK (http://www.vmtk.org/).

#include <dolfin.h>
#include "Poisson.h"

using namespace dolfin;

int main()
{
  // Create mesh and finite element
  auto mesh = std::make_shared<Mesh>("../aneurysm.xml.gz");

  // Define variational problem
  Constant f(0.0);
  auto V = std::make_shared<Poisson::FunctionSpace>(mesh);
  Poisson::BilinearForm a(V, V);
  Poisson::LinearForm L(V);
  L.f = f;

  // Define boundary condition values
  auto u0 = std::make_shared<Constant>(0.0);
  auto u1 = std::make_shared<Constant>(1.0);
  auto u2 = std::make_shared<Constant>(2.0);
  auto u3 = std::make_shared<Constant>(3.0);

  // Define boundary conditions
  DirichletBC bc0(V, u0, 0);
  DirichletBC bc1(V, u1, 1);
  DirichletBC bc2(V, u2, 2);
  DirichletBC bc3(V, u3, 3);
  std::vector<const DirichletBC*> bcs{{&bc0, &bc1, &bc2, &bc3}};

  // Set PETSc MUMPS paramter (this is required to prevent a memory
  // error in some cases when using MUMPS LU solver).
  #ifdef HAS_PETSC
  PETScOptions::set("mat_mumps_icntl_14", 40.0);
  #endif

  // Compute solution
  Function u(V);
  solve(a == L, u, bcs);

  // Write solution to file
  File file("u.pvd");
  file << u;

  // Plot solution
  plot(u);
  interactive();

  return 0;
}
