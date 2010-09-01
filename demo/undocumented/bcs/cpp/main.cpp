// Copyright (C) 2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-05-23
// Last changed: 2010-09-02
//
// This demo illustrates how to set boundary conditions for meshes
// that include boundary indicators. The mesh used in this demo was
// generated with VMTK (http://villacamozzi.marionegri.it/~luca/vmtk/).

#include <dolfin.h>
#include "Poisson.h"
#include <boost/assign/list_of.hpp>

using namespace dolfin;

int main()
{
  // Create mesh and finite element
  Mesh mesh("aneurysm.xml.gz");

  // Define variational problem
  Constant f(0.0);
  Poisson::FunctionSpace V(mesh);
  Poisson::BilinearForm a(V, V);
  Poisson::LinearForm L(V);
  L.f = f;

  // Define boundary condition values
  Constant u0(0.0);
  Constant u1(1.0);
  Constant u2(2.0);
  Constant u3(3.0);

  // Define boundary conditions
  DirichletBC bc0(V, u0, 0);
  DirichletBC bc1(V, u1, 1);
  DirichletBC bc2(V, u2, 2);
  DirichletBC bc3(V, u3, 3);
  std::vector<const BoundaryCondition*> bcs = boost::assign::list_of(&bc0)(&bc1)(&bc2)(&bc3);

  // Solve PDE and plot solution
  VariationalProblem pde(a, L, bcs);
  Function u(V);
  pde.solve(u);
  plot(u);

  return 0;
}
