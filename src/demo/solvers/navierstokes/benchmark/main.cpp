// Copyright (C) 2005 Johan Hoffman.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg 2005.
//
// First added:  2002-11-29
// Last changed: 2005-11-29
//
// A solver for the Navier-Stokes equations 
//
//     du/dt + u * grad u - nu * div grad u + grad p = f 
//     div u = 0 

#include <dolfin.h>

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

// Force term
class ForceFunction_3D : public Function
{
  real eval(const Point& p, unsigned int i)
  {
    if (i==0) return 0.0;
    if (i==1) return 0.0;
    if (i==2) return 0.0;
    dolfin_error("Wrong vector component index");
    return 0.0;
  }
};

// Boundary condition for momentum equation 
class BC_Momentum_3D : public BoundaryCondition
{
  // These are boundary conditions for the flow past a 
  // circular cylinder in 3d. We use a uniform unit inflow 
  // velocity, no slip bc on the cylinder, and slip bc 
  // at the lateral boundaries: 
  // 
  // u = (u_1,u_2,u_3) = (1,0,0) at x = 0
  // u_2 = 0 for y = 0 and y = 1.4
  // u_3 = 0 for z = 0 and z = 0.4 
  // u = 0 on the cylinder with radie 0.05 and center at (0.5,0.7,z)
  // 

  const BoundaryValue operator() (const Point& p, unsigned int i)
  {
    real bmarg = 1.0e-3;

    BoundaryValue value;
    if (i==0){
      if ( p.x < (bmarg + DOLFIN_EPS)){
	value.set(1.0);
      }      
      if ( sqrt(sqr(p.x - 0.5) + sqr(p.y - 0.7)) < (0.05 + (bmarg + DOLFIN_EPS))){
	value.set(0.0);
      }       
    } else if (i==1){
      if ( (p.y < (bmarg + DOLFIN_EPS)) || (p.y > (1.4 - (bmarg + DOLFIN_EPS))) ){
	value.set(0.0);
      }
      if ( p.x < (bmarg + DOLFIN_EPS)){
	value.set(0.0);
      }
      if ( sqrt(sqr(p.x - 0.5) + sqr(p.y - 0.7)) < (0.05 + (bmarg + DOLFIN_EPS))){
	value.set(0.0);
      }       
    } else if (i==2){
      if ( sqrt(sqr(p.x - 0.5) + sqr(p.y - 0.7)) < (0.05 + (bmarg + DOLFIN_EPS))){
	value.set(0.0);
      } 
      if (p.x < (bmarg + DOLFIN_EPS)){
	value.set(0.0);
      }
      if ( (p.z < (bmarg + DOLFIN_EPS)) || (p.z > (0.4 - (bmarg + DOLFIN_EPS))) ){
	value.set(0.0);
      }
    } else{
      dolfin_error("Wrong vector component index");
    }
  
    return value;
  }
};

// Boundary condition for continuity equation 
class BC_Continuity_3D : public BoundaryCondition
{
  // This is an approximation of the outflow boundary condition: 
  // 
  // nu * du/dn - np = 0 
  // 
  // Assuming the viscosity nu is small, we may approximate 
  // this boundary condition with zero pressure at outflow. 
  
  const BoundaryValue operator() (const Point& p)
  {
    real bmarg = 1.0e-3;

    BoundaryValue value;
    if (p.x > (2.1 - (bmarg + DOLFIN_EPS)))
      value.set(0.0);
    
    return value;
  }
};

// Boundary condition for momentum equation 
class BC_Momentum_2D : public BoundaryCondition
{
  // These are boundary conditions for the flow past a 
  // circular cylinder in 2d. 
  // This is the  benchmark problem: 2D-3, defined in  
  // http://www.mathematik.uni-dortmund.de/~featflow/ture/paper/benchmark_results.ps.gz

  const BoundaryValue operator() (const Point& p, unsigned int i)
  {
    real bmarg = 1.0e-3;

    BoundaryValue value;
    if (i==0){
      if ( p.x < (0.0 + DOLFIN_EPS + bmarg)){
	value.set( (1.0/sqr(0.41)) * sin(DOLFIN_PI*time()*0.125) * 6.0*p.y*(0.41-p.y) );
      } 
      if ( p.y < (0.0 + DOLFIN_EPS + bmarg)){
	value.set(0.0);
      } 
      if ( p.y > 0.41 - DOLFIN_EPS - bmarg){
	value.set(0.0);
      } 
      if ( sqrt(sqr(p.x - 0.2) + sqr(p.y - 0.2)) < (0.05 + DOLFIN_EPS + bmarg)){
	value.set(0.0);
      }       
    } else if (i==1){
      if ( p.x < (0.0 + DOLFIN_EPS + bmarg)){
	value.set(0.0);
      } 
      if ( p.y < (0.0 + DOLFIN_EPS + bmarg)){
	value.set(0.0);
      } 
      if ( p.y > 0.41 - DOLFIN_EPS - bmarg){
	value.set(0.0);
      } 
      if ( sqrt(sqr(p.x - 0.2) + sqr(p.y - 0.2)) < (0.05 + DOLFIN_EPS + bmarg)){
	value.set(0.0);
      }       
    } else{
      dolfin_error("Wrong vector component index");
    }
  
    return value;
  }
};

// Boundary condition for continuity equation 
class BC_Continuity_2D : public BoundaryCondition
{
  // This is an approximation of the outflow boundary condition: 
  // 
  // nu * du/dn - np = 0 
  // 
  // Assuming the viscosity nu is small, we may approximate 
  // this boundary condition with zero pressure at outflow. 
    
  const BoundaryValue operator() (const Point& p)
  {
    real bmarg = 1.0e-3;

    BoundaryValue value;
    if (p.x > (2.2 - DOLFIN_EPS - bmarg)){
      value.set(0.0);
    }
    
    return value;
  }
};

int main(int argc, char* argv[])
{
  dolfin_init(argc, argv);


  /*  
  // This is the 2d benchmark problem: 2D-3, defined in  
  // http://www.mathematik.uni-dortmund.de/~featflow/ture/paper/benchmark_results.ps.gz
  Mesh mesh("cylinder_2d_bmk.xml.gz");
  BC_Momentum_2D bc_mom;
  BC_Continuity_2D bc_con;
  ForceFunction_2D f;
  */


  // This is a 3d benchmark problem with Re = 3900, described in 
  // http://www.nada.kth.se/~jhoffman/archive/papers/cc.pdf
  Mesh mesh("cylinder_3d_bmk.xml.gz");
  BC_Momentum_3D bc_mom;
  BC_Continuity_3D bc_con;
  ForceFunction_3D f;

  NSESolver::solve(mesh, f, bc_mom, bc_con); 
  
  return 0;
}
