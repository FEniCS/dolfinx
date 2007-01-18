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
//#include <dolfin/ALEBoundaryCondition.h>
//#include <dolfin/ALESolver.h>
#include <dolfin/ALEFunction.h>

using namespace dolfin;






// Force term
class ForceFunction_2D : public Function
{
  real eval(const Point& p, unsigned int i)
  {
    if (i==0) return 0.0;
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

     if (i==0) {
       if (m.y < 0.5) {   //fluid

	 if (m.x < (2.0 - DOLFIN_EPS - bmarg))
	   value.set(0.0);

	 if (m.x < (0.0 + DOLFIN_EPS + bmarg))
	   value.set((0.5 - m.y) * m.y * 20);

       } else {            //structure
	 value.set(0.0);
       }
     } else if (i==1) {
       
       value.set(0.0);
     }
     
  }
};
// Boundary condition for momentum equation 

// Boundary condition for continuity equation 
class BC_Continuity_2D : public ALEBoundaryCondition
{
  // This is an approximation of the outflow boundary condition: 
  // 
  // nu * du/dn - np = 0 
  // 
  // Assuming the viscosity nu is small, we may approximate 
  // this boundary condition with zero pressure at outflow. 
    
  void eval(BoundaryValue& value, const Point& p, const Point& m, unsigned int i)
  {
    real bmarg = 1.0e-3;

    if (m.y < 0.5)   //fluid
      if (p.x > (2.0 - DOLFIN_EPS - bmarg))
	value = 0.0;
  }
};


class BisectionFunction : public Function
{
  real eval(const Point& p, unsigned int i)
  {
    if (p.y > 0.5 ) return 0; // structure
    return 1;      //fluid

    //return 1; // its all fluid
  }
};

//---------------------------------------------------------------------
int main(int argc, char* argv[])
{
  dolfin_init(argc, argv);

  // mesh rectangle [0,0] - [2,1]
  Mesh mesh("rect_ale_ns.xml");
 
  //need to refine mesh
  for (int grain = 0; grain < 4; grain++) 
    mesh.refineUniformly();
  
  BC_Momentum_2D    bc_mom;
  BC_Continuity_2D  bc_con;
  ForceFunction_2D  f;
  BisectionFunction bisect;
 
  real rhof = 1;
  real rhos = 1000;
  real k    = 1e-4;
  real E    = 20;
  real elnu = 0.3;
  
  
  FSISolver::solve(mesh, f, bc_mom, bc_con, bisect, rhof, rhos, E, elnu, k); 
  return 0;
}
