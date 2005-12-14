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
class ForceFunction : public Function
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

// Initial solution 
class InitialSolution : public Function
{
  real eval(const Point& p, unsigned int i)
  {
    if (i==0){
      if (p.y < 0.2) 
	return 5.0*p.y;
      else           
	return 1.0;
    }
    if (i==1){
      if ( (p.y < 0.2) && (fabs(p.z-0.5) < 0.25) )
	return - 0.05 * ( - cos(2.0*2.0*DOLFIN_PI*p.z) * sin(5.0*DOLFIN_PI*p.y) );
      else 
	return 0.0;
    }  
    if (i==2){
      if ( (p.y < 0.2) && (fabs(p.z-0.5) < 0.25) )
	return - 0.05 * ( - sin(2.0*2.0*DOLFIN_PI*p.z) * cos(5.0*DOLFIN_PI*p.y) );
      else 
	return 0.0;
    }
    dolfin_error("Wrong vector component index");
    return 0.0;
  }
};


/*
// Boundary condition for momentum equation 
class BC_Momentum : public BoundaryCondition
{
  const BoundaryValue operator() (const Point& p, unsigned int i)
  {
    BoundaryValue value;
    if (i==0){
      if (fabs(p.x - 0.0) < DOLFIN_EPS){
	if (p.y < 0.2)
	  value.set(5.0*p.y);
	else 
	  value.set(1.0);
      }
      if (fabs(p.y - 0.0) < DOLFIN_EPS)  
	value.set(0.0);
    } else if (i==1){
      if (fabs(p.x - 0.0) < DOLFIN_EPS){
	if ( (p.y < 0.2) && (fabs(p.z-0.5) < 0.25) )
	  value.set(- 0.05 * ( - cos(2.0*2.0*DOLFIN_PI*p.z) * sin(5.0*DOLFIN_PI*p.y) ));
	else 
	  value.set(0.0);
      }
      if ( (fabs(p.y - 0.0) < DOLFIN_EPS) || (fabs(p.y - 1.0) < DOLFIN_EPS) ) 
	value.set(0.0);
    } else if (i==2){
      if (fabs(p.x - 0.0) < DOLFIN_EPS){
	if ( (p.y < 0.2) && (fabs(p.z-0.5) < 0.25) )
	  value.set(- 0.05 * ( - sin(2.0*2.0*DOLFIN_PI*p.z) * cos(5.0*DOLFIN_PI*p.y) ));
	else 
	  value.set(0.0);
      }
      if ( (fabs(p.z - 0.0) < DOLFIN_EPS) || (fabs(p.z - 1.0) < DOLFIN_EPS) ) 
	value.set(0.0);
    } else{
      dolfin_error("Wrong vector component index");
    }
  
    return value;
  }
};

// Boundary condition for continuity equation 
class BC_Continuity : public BoundaryCondition
{
  const BoundaryValue operator() (const Point& p)
  {
    BoundaryValue value;
    if (fabs(p.x - 12.0) < DOLFIN_EPS)
      value.set(0.0);
    
    return value;
  }
};
*/

// Boundary condition for momentum equation 
class BC_Momentum_3D : public BoundaryCondition
{
  
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
      if ( (p.y < (bmarg + DOLFIN_EPS)) || (p.y > (1.4 - (bmarg + DOLFIN_EPS))) ||  
	   (p.z < (bmarg + DOLFIN_EPS)) || (p.z > (0.4 - (bmarg + DOLFIN_EPS))) ){
	value.set(1.0);
      }      
    } else if (i==1){
      if ( p.x < (bmarg + DOLFIN_EPS)){
	value.set(0.0);
      }
      if ( (p.y < (bmarg + DOLFIN_EPS)) || (p.y > (1.4 - (bmarg + DOLFIN_EPS))) ){
	value.set(0.0);
      }
      if ( sqrt(sqr(p.x - 0.5) + sqr(p.y - 0.7)) < (0.05 + (bmarg + DOLFIN_EPS))){
	value.set(0.0);
      }       
      if ( (p.y < (bmarg + DOLFIN_EPS)) || (p.y > (1.4 - (bmarg + DOLFIN_EPS))) ||
	   (p.z < (bmarg + DOLFIN_EPS)) || (p.z > (0.4 - (bmarg + DOLFIN_EPS))) ){
	value.set(0.0);
      }      
    } else if (i==2){
      if (p.x < (bmarg + DOLFIN_EPS)){
	value.set(0.0);
      }
      if ( (p.z < (bmarg + DOLFIN_EPS)) || (p.z > (0.4 - (bmarg + DOLFIN_EPS))) ){
	value.set(0.0);
      }
      if ( sqrt(sqr(p.x - 0.5) + sqr(p.y - 0.7)) < (0.05 + (bmarg + DOLFIN_EPS))){
	value.set(0.0);
      }       
      if ( (p.y < (bmarg + DOLFIN_EPS)) || (p.y > (1.4 - (bmarg + DOLFIN_EPS))) ||
	   (p.z < (bmarg + DOLFIN_EPS)) || (p.z > (0.4 - (bmarg + DOLFIN_EPS))) ){
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

public:
  BC_Momentum_2D::BC_Momentum_2D() : BoundaryCondition()
  {
  }

  const BoundaryValue operator() (const Point& p, unsigned int i)
  {
    BoundaryValue value;
    if (i==0){
      if ( p.x < 0.0 + 0.01 + DOLFIN_EPS){
	value.set( (1.0/sqr(0.41)) * sin(DOLFIN_PI*time()*0.125) * 6.0*p.y*(0.41-p.y) );
      } 
      if ( p.y < 0.0 + 0.01 + DOLFIN_EPS){
	value.set(0.0);
      } 
      if ( p.y > 0.41 - 0.01 - DOLFIN_EPS){
	value.set(0.0);
      } 
      if ( sqrt(sqr(p.x - 0.2) + sqr(p.y - 0.2)) < 0.051 + DOLFIN_EPS){
	value.set(0.0);
      }       
    } else if (i==1){
      if ( p.y < 0.0 + 0.01 + DOLFIN_EPS){
	value.set(0.0);
      } 
      if ( p.y > 0.41 - 0.01 - DOLFIN_EPS){
	value.set(0.0);
      } 
      if (sqrt(sqr(p.x - 0.2) + sqr(p.y - 0.2)) < 0.051 + DOLFIN_EPS){
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
  const BoundaryValue operator() (const Point& p)
  {
    BoundaryValue value;
    if (fabs(p.x - 2.2) < DOLFIN_EPS){
      value.set(0.0);
    }
    
    return value;
  }
};

int main(int argc, char* argv[])
{
  dolfin_init(argc, argv);

  //Mesh mesh("cylinder_2d_bmk.xml.gz");
  Mesh mesh("cylinder_3d_bmk.xml.gz");
  ForceFunction f;

  /*
  BC_Momentum_2D bc_mom;
  BC_Continuity_2D bc_con;
  */

  BC_Momentum_3D bc_mom;
  BC_Continuity_3D bc_con;
  
  NSESolver::solve(mesh, f, bc_mom, bc_con); 
  
  return 0;
}
