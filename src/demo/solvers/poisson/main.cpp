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

// Source term
// Example using NewFunction

class f : public NewFunction
{
public:
  f(const Mesh& mesh, const NewFiniteElement& element, NewVector& x) :
    NewFunction(mesh, element, x)
  {
  }
  virtual real operator()(const Point& p)
  {
    real pi = DOLFIN_PI;
    return 14.0 * pi*pi * sin(pi*p.x) * sin(2.0*pi*p.y) * sin(3.0*pi*p.z);
  }
};

/*
real f(real x, real y, real z, real t)
{
  real pi = DOLFIN_PI;
  return 14.0 * pi*pi * sin(pi*x) * sin(2.0*pi*y) * sin(3.0*pi*z);
}
*/

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
  MyBC bc;
  PoissonSolver::solve(mesh, bc);
  
  return 0;
}
