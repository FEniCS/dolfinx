// Copyright (C) 2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-05-23
// Last changed: 2008-05-23
//
// This demo illustrates how to set boundary conditions for meshes
// that include boundary indicators. The mesh used in this demo was
// generated with VMTK (http://villacamozzi.marionegri.it/~luca/vmtk/).

#include <dolfin.h>
#include "Poisson.h"

using namespace dolfin;

int main()
{
  // Create mesh and finite element
  Mesh mesh("../../../../data/meshes/aneurysm.xml.gz");

  // Define variational problem
  Function f(mesh, 0.0);
  PoissonBilinearForm a;
  PoissonLinearForm L(f);

  // Define boundary condition values
  Function u0(mesh, 0.0);
  Function u1(mesh, 1.0);
  Function u2(mesh, 2.0);
  Function u3(mesh, 3.0);

  // Define boundary conditions
  DirichletBC bc0(u0, mesh, 0);
  DirichletBC bc1(u1, mesh, 1);
  DirichletBC bc2(u2, mesh, 2);
  DirichletBC bc3(u3, mesh, 3);
  Array<BoundaryCondition*> bcs(&bc0, &bc1, &bc2, &bc3);

  // Solve PDE and plot solution
  LinearPDE pde(a, L, mesh, bcs);
  Function u;
  pde.solve(u);
  plot(u);

  return 0;
}
