// Copyright (C) 2004-2005 Johan Jansson.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg 2005.
//
// First added:  2004
// Last changed: 2005-12-28

#include <dolfin.h>

using namespace dolfin;

// Density
class Density : public Function
{
  real eval(const Point& p, unsigned int i)
  {
    if(p.x < 0.01 && p.y > 0.0)
      return 1.0e3;
    else
      return 1.0e3;
  }
};

// Right-hand side
class Source : public Function
{
  real eval(const Point& p, unsigned int i)
  {
    if(i == 1)
      return -10.0 * 1e3;
    else
      return 0.0;
  }
};

// Initial velocity
class InitialVelocity : public Function
{
  real eval(const Point& p, unsigned int i)
  {
    real result = 0.0;

    return result;
  }
};

// Boundary condition
class MyBC : public BoundaryCondition
{
  void eval(BoundaryValue& value, const Point& p, unsigned int i)
  {
    if(p.x == 0.0)
      value = 0.0;
  }
};

int main(int argc, char **argv)
{
  dolfin_output("plain text");

#ifdef HAVE_PETSC_H

  real T = 5.0;  // final time
  real k = 1.0e-3; // time step

  set("ODE method", "cg");
  set("ODE order", 1);

  set("ODE save solution", false);
  set("ODE solution file name", "primal.py");
  set("ODE number of samples", 400);

  set("ODE fixed time step", true);
  set("ODE initial time step", k);
  set("ODE maximum time step", k);
  set("ODE maximum iterations", 100);

  Mesh mesh("tetmesh-1c.xml.gz");

  Source f;
  Density rho;
  InitialVelocity v0;
  MyBC bc;

  real E = 1.5 * 5.0e5; // Young's modulus
  real nu = 0.3; // Poisson's ratio
  real nuv = 0.0 * 1.0e2; // viscosity
  real nuplast = 0.0; // plastic viscosity


  ElasticityUpdatedSolver::solve(mesh, f, v0, rho, E, nu, nuv,
				 nuplast, bc, k, T);
#else

  cout << "DOLFIN must be configured with PETSc to run this demo" << endl; 

#endif

  return 0;
}

