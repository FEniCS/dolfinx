// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// A simple test program for the Poisson solver, solving
//
//     - div grad u(x,y,z) = f(x,y,z)
//
// on the unit cube with the source f given by
//
//     f(x,y,z) = 14 pi^2 sin(pi x) sin(2pi y) sin(3pi z).
//
// and homogeneous Dirichlet boundary conditions.
// This problem has the exact solution
//
//     u(x,y,z) = sin(pi x) sin(2pi y) sin(3pi z).

#include <dolfin/PoissonSolver.h>
#include <dolfin/NewFunction.h>
#include <dolfin/Mesh.h>
#include <dolfin.h>

using namespace dolfin;

// Right-hand side
class MyFunction : public NewFunction
{
  real operator() (const Point& p) const
  {
    return 8.0;
  }
};

// Boundary condition
class MyBC : public NewBoundaryCondition
{
  const BoundaryValue operator() (const Point& p)
  {
    BoundaryValue value;
    if ( (fabs(p.x - 0.0) < DOLFIN_EPS) || (fabs(p.x - 1.0) < DOLFIN_EPS ) )
      value.set(0.0);
    
    return value;
  }
};

int main()
{
  Mesh mesh("mesh.xml.gz");
  MyFunction f;
  MyBC bc;
  
  PoissonSolver::solve(mesh, f, bc);
  
  return 0;
}
