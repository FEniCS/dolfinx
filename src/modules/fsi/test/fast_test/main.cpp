// Copyright (C) 2005 Johan Hoffman.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg 2005.
//
// First added:  2002-11-29
// Last changed: 2005-12-28
//
// A cG(1)cG(1) solver for the incompressible Navier-Stokes equations 
//
//     du/dt + u * grad u - nu * div grad u + grad p = f 
//     div u = 0 

#include <dolfin.h>
#include <dolfin/dolfin_modules.h>

using namespace dolfin;


// Defines the fluid domain, conversely !FluidDomain is structure
// domain.
bool FluidDomain(const Point& r)   
{
  return (0.2 < r.y() && r.y() < 0.8);
}

bool FluidDomainContainer(const Point& r)
{
  return (0.1 > r.y() || r.y() > 0.9);
}
//---------------------------------------------------------------------
// Force term
class ForceFunction_2D : public Function
{
  real eval(const Point& p, unsigned int i)
  {
    return 0.0;
  }
};
//---------------------------------------------------------------------
// Boundary condition for momentum equation 
class BC_Momentum_2D : public ALEBoundaryCondition
{
  real amp() 
  {
    
    real pi = 3.14;
    
    return (1.0+0.5*sin(4.0*pi*time())+0.1*sin(2.0*pi*time()));
  }
  //-------------------------------------------------------------------
  void eval(BoundaryValue& value, const Point& p, const Point& r, unsigned int i)
  {
     real bmarg = 1.0e-3;
 
     // if (p.x() < (0.0 + DOLFIN_EPS + bmarg))
     //   value.set(0.0);

     if (!FluidDomain(p)) {    // structure
       // if (p.x() < (0.0 + DOLFIN_EPS + bmarg) ||  // fixed at ends
       //     p.x() > (2.0 - DOLFIN_EPS - bmarg))
       value.set(0.0);
     } else {                                     // fluid
       if (i == 0)
	 if (p.x() < (0.0 + DOLFIN_EPS + bmarg))  // inflow
	   value.set(amp()*(0.8 - p.y()) * (p.y() - 0.2) * 4);
     }
  }
};
//---------------------------------------------------------------------
// Boundary condition for continuity equation 
class BC_Continuity_2D : public ALEBoundaryCondition
{
  void eval(BoundaryValue& value, const Point& p, const Point& r, unsigned int i)
  {
    real bmarg = 1.0e-3;

    if (p.x() > (2.0 - DOLFIN_EPS - bmarg))
      if (FluidDomain(p))
	value.set(0.0);
  }
};
//---------------------------------------------------------------------
class BisectionFunction : public Function
{
  real eval(const Point& r, unsigned int i)
  {
    if (FluidDomain(r)) return 1; // fluid
    if (FluidDomainContainer(r)) return 1;
    return 0;                     // structure
  }
};
//---------------------------------------------------------------------
int main(int argc, char* argv[])
{
  dolfin_init(argc, argv);

  // mesh rectangle [0,0] - [2,1]
  Mesh mesh("rect_ale_ns.xml");

  //need to refine mesh
  for (int grain = 0; grain < 6; grain++) 
    mesh.refine();
  
  BC_Momentum_2D    bc_mom;  
  BC_Continuity_2D  bc_con;
  ForceFunction_2D  f;
  BisectionFunction bisect;
 
  real rhof = 1;          // fluid:     density
  real nu   = 1.0/3900.0; // fluid:     viscosity 
  real rhos = 1;          // structure: density
  real E    = 40;         // structure: Young's modulus
  real elnu = 0.3;        // structure: Poisson's ratio
  real T    = 2.0;        // final time
  real k    = 1e-2;       // time step size
 
  FSISolver::solve(mesh, f, bc_mom, bc_con, bisect, rhof, rhos, E, elnu, nu, T, k); 
  return 0;
}
