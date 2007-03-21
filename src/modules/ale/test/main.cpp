// Copyright (C) 2005 Johan Hoffman.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg 2005.
//
// First added:  2002-11-29
// Last changed: 2005-12-28
//
// A cG(1)cG(1) solver for the incompressible Navier-Stokes ALE equations 
//
//     du/dt + u * grad u - nu * div grad u + grad p = f 
//     div u = 0 

#include <dolfin.h>
#include <dolfin/dolfin_modules.h>


using namespace dolfin;



// This is the external force function which is 
// evaluated on the boundary of the mesh.
class ALEExtFunction : public ALEFunction
{
  real eval(const Point& p, const Point& r, unsigned int i)
  {

    if (i == 1) {
      if (p.y() > 0.5)
	return 0.08*sin(3*time())*(p.x()*(2-p.x()));
      else 
	return (-1)*0.08*sin(3*time())*(p.x()*(2-p.x()));
    }
    return 0;
      
  }	
};


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

  void eval(BoundaryValue& value, const Point& p, const Point& r, unsigned int i)
  {
     real bmarg = 1.0e-3;
  
     if (i==0){  // x direction
       if ( p.x() < (2.0 - DOLFIN_EPS - bmarg))
	 value.set(0.0 + w->eval(p,r,i));
       if ( r.x() < (0.0 + DOLFIN_EPS + bmarg)){
	 value.set((1 - r.y()) * r.y() * 4 + w->eval(p,r,i));
       } 
     
    } else if (i==1){  // y direction
      if ( p.x() < (2.0 - DOLFIN_EPS - bmarg))
	value.set(0.0 + w->eval(p,r,i));
    } else{
      dolfin_error("Wrong vector component index");
    }
  }
};

// Boundary condition for continuity equation 
class BC_Continuity_2D : public ALEBoundaryCondition
{
  void eval(BoundaryValue& value, const Point& p, const Point& r, unsigned int i)
  {
    real bmarg = 1.0e-3;

    if (p.x() > (2.0 - DOLFIN_EPS - bmarg))
      value = 0.0;
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
    mesh.refine();
  
  BC_Momentum_2D   bc_mom;
  BC_Continuity_2D bc_con;
  ForceFunction_2D f;
  ALEExtFunction   e;
  
  ALESolver::solve(mesh, f, bc_mom, bc_con, e); 
  
  return 0;
}
