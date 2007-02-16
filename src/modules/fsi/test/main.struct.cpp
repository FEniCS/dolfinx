// Copyright (C) 2005 Johan Hoffman.
// LiceALEd under the GNU GPL Version 2.
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
#include <dolfin/ALEFunction.h>

using namespace dolfin;

// Force term
class ForceFunction_2D : public Function
{
  real eval(const Point& p, unsigned int i)
  {
    if (i==0) return 5.0;
    if (i==1) return 0.0;
    dolfin_error("Wrong vector component index");
    return 0.0;
  }
};


// Boundary condition for momentum equation 
class BC_Momentum_2D : public ALEBoundaryCondition
{
  void eval(BoundaryValue& value, const Point& p, const Point& m, unsigned int i)
  {
     real bmarg = 1.0e-3;

     if (p.x < (0.0 + DOLFIN_EPS + bmarg))
       value.set(0.0);
  }
};

// Boundary condition for continuity equation 
class BC_Continuity_2D : public ALEBoundaryCondition
{

  void eval(BoundaryValue& value, const Point& p, const Point& m, unsigned int i)
  {

  }
};


class BisectionFunction : public Function
{
  real eval(const Point& p, unsigned int i)
  {
    return 0; // its all structure
  }
};

//---------------------------------------------------------------------
int main(int argc, char* argv[])
{
  dolfin_init(argc, argv);
  UnitSquare mesh(1,1);

  for (int grain = 0; grain < 1; grain++) 
    mesh.refineUniformly();
  
  BC_Momentum_2D    bc_mom;
  BC_Continuity_2D  bc_con;
  ForceFunction_2D  f;
  BisectionFunction bisect;

  real rhof = 1;
  real rhos = 1;
  real k    = 1e-4;
  real E    = 20;
  real elnu = 0.3;
  
  
  FSISolver::solve(mesh, f, bc_mom, bc_con, bisect, rhof, rhos, E, elnu, k); 
  
  return 0;
}
