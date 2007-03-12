// Copyright (C) 2005 Johan Hoffman.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg 2005.


#include <dolfin.h>
#include <dolfin/init.h>
//#include "dolfin/CNSmix2D.h"
#include "dolfin/CNSSolver.h"
//#include "dolfin/CNSmix3D.h"

using namespace dolfin;

// Initial data
class InitialData_2D : public Function
{

  /*
  // Smooth initial data
  real eval(const Point& p, unsigned int i)
  {
    if (i==0)
      {
	// Rho
 	return  sqr(1 - p.x()); // * sqr(1 + p.x()); //sin(p.x()) * cos(p.y());
      }
    else if (i==1)
      {
	// m0
	return 0.0;
      }
    else if (i==2)
      {
	// m1
	return 0.0;
      }
    else if (i==3)
      {
 	return  2.0 * sqr(1 - p.x()) * sqr(1 + p.x()); //sin(p.x()) * cos(p.y());
	//return 2.0; //sin(p.x());
      }
    else
      {
	dolfin_error("Wrong vector component index");
	return 0.0;
      }

  }
  */
  
  // Discontinuous initial data
  
  real eval(const Point& p, unsigned int i)
  {
    if (i==0)
      {
	// Rho
 	if(p.x() < 0.5)
	  // 	if(true || p.x() < 0.5)
	  {
	    return 1.0;
	  }
	else
	  {
	    return 0.125;
	  }
      }
    else if (i==1)
      {
	// m0
	return 0.0;
	//	return 2.0;
      }
    else if (i==2)
      {
	// m1
	return 0.0;
	//	return 3.0;
      }
    else if (i==3)
      {
	// e
 	if(p.x() < 0.5)
	  // 	if(true || p.x() < 0.5)
	  {
 	    return 2.0;
	    // 	    return 4.0;
	  }
	else
	  {
	    return 0.2;
	  }
      }
    else
      {
	dolfin_error("Wrong vector component index");
      return 0.0;
      }
  }
  
};

// Force term
class ForceFunction_2D : public Function
{
  real eval(const Point& p, unsigned int i)
  {
    if (i==0) return 0.0;
    if (i==1) return 0.0;
    if (i==2) return 0.0;
    if (i==3) return 0.0;
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
    if (i==3) return 0.0;
    if (i==4) return 0.0;
    dolfin_error("Wrong vector component index");
    return 0.0;
  }
};


// Boundary condition for equation 

class BC_CNS_2D : public BoundaryCondition
{
  // set the boundary conditions: i=0 for rho, i=1,2,3 for m, i=4 for e
  void eval(BoundaryValue& value, const Point& p, unsigned int i)
  {
    if (i==0)
      {
	// Rho
	/* if(p.x() < 0.5)
	  {
	    value.set(1.0);
	  }
	else
	  {
	    value.set(0.125);
	    }
	*/
	//value.set(0.0);
      }
    else if (i==1)
      {

	real bmarg = 1.0e-3;
	if ( p.x() < (bmarg + DOLFIN_EPS)){
	  value.set(0.0);
	}

	if ( p.x() > (1 -( bmarg + DOLFIN_EPS ))){
	  value.set(0.0);
	}


	// m0
        //value.set(0.0);
      }
    else if (i==2)
      {
	real bmarg = 1.0e-3;
	if ( p.y() < (bmarg + DOLFIN_EPS)){
	  value.set(0.0);
	}

	if ( p.y() > (1 -( bmarg + DOLFIN_EPS ))){
	  value.set(0.0);
	}

	// m1
	//value.set(0.0);
	}
    else if (i==3)
      {
	/*// e
	if(p.x() < 0.5)
	  {
	    value.set(2.0);
	  }
	else
	  {
	    value.set(0.2);
	    }*/
      }
    
    else 
      {
	dolfin_error("Wrong vector component index");
      }

  }
};

class BC_CNS_3D : public BoundaryCondition
{
  // set the boundary conditions: i=0 for rho, i=1,2,3 for m, i=4 for e
  void eval(BoundaryValue& value, const Point& p, unsigned int i)
  {
    value.set(0.0);
/*
    real bmarg = 1.0e-3;
      if ( p.x < (bmarg + DOLFIN_EPS)){
	value.set(0.0);
      }      
      if ( p.y < (bmarg + DOLFIN_EPS)){
	value.set(0.0);
      }      
      if ( p.z < (bmarg + DOLFIN_EPS)){
	value.set(0.0);
      }    
*/ 
  }
};

int main(int argc, char* argv[])
{
  dolfin_init(argc, argv);

  //UnitSquare  mesh(120, 120);
  //UnitSquare  mesh(40, 40);
  UnitSquare  mesh(50,50);
  
  //    mesh.refine();
  // mesh.refine();
  //  mesh.refine();

  BC_CNS_2D bc;
  ForceFunction_2D f;
  InitialData_2D w0;

  CNSSolver::solve(mesh, f, w0, bc); 
  
  return 0;
}

