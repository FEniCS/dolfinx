// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin.h>

using namespace dolfin;

// Source term
real f(real x, real y, real z, real t, int i)
{
  if(i == 0)
  {
    if(t > 0.0 && t < 0.5 && x > 0.99)
    {
      return -10.0;
    }
    else if(t > 0.0 && t < 0.5 && x < 0.01)
    {
      return 10.0;
    }
  }

  /*
  real lambda = 1.0;
  real mu = 1.0;

  real force = 0;

  Vector axis(3), center(3), p(3), vforce(3);

  if(x == 1.0)
  {
    p(0) = x; p(1) = y; p(2) = z;

    axis(0) = 1; axis(1) = 0; axis(2) = 0;
    center(0) = 0; center(1) = 0.5; center(2) = 0.5;
    
    p.add(-1, center);

    axis.cross(p, vforce);

    //dolfin_debug("foo");
    //p.show();
    //axis.show();
    //vforce.show();

    vforce *= 10;

    
  }

  if(i == 0)
  {
    //return (lambda + mu) * (1 - 2 * x) * (1 - 2 * y);
    //if(t <= 5.0)
    //return 4.0;
    //else
    
    //if(t < 1.0)
    //force += 1.0;
    //if(t < 3.0)
    //force += 0.2;
    //if(x == 1.0 && t > 5.0 && t < 5.0)
    //  return vforce(0);
    //if(x == 1.0 && t > 5.0 && t < 5.0)
    //  return vforce(0);
  }
  else if(i == 1)
  {
    //return -2 * mu * y * (1 - y) - 2 * (lambda + 2 * mu) * x * (1 - x);
    
    //if(t > 5.0 && t < 5.5 && x == 1.0)
    //force += 10.0;
    //force += -0.01;
    //if(x == 1.0 && t > 5.0 && t < 5.0)
    //  return vforce(1);
    if(x == 1.0 && z == 1.0)
      return -5.0;
    else if(x == 1.0 && z == 0.0)
      return 5.0;
  }
  else if(i == 2)
  {
    //return -2 * mu * y * (1 - y) - 2 * (lambda + 2 * mu) * x * (1 - x);
    
    //if(x == 1.0 && t > 5.0 && t < 5.0)
    //  return vforce(2);
    if(x == 1.0 && y == 1.0)
      return 5.0;
    else if(x == 1.0 && y == 0.0)
      return -5.0;
  }
    
*/

  return 0.0;
}

// Diffusivity
real a(real x, real y, real z, real t)
{
  //return 0.2;
  return 0.0;
}

// Convection
real b(real x, real y, real z, real t, int i)
{
  //if ( i == 0 )
  //return -5.0;

  return 0.0;
}

// Boundary conditions
void mybc(BoundaryCondition& bc)
{
  //bc.set(BoundaryCondition::DIRICHLET, 0.0, 0);
  
  /*
  if ( bc.coord().x == 0.0 )
  {
    //bc.set(BoundaryCondition::DIRICHLET, 0.0, 0);
    //bc.set(BoundaryCondition::DIRICHLET, 0.0, 1);
    //bc.set(BoundaryCondition::DIRICHLET, 0.0, 2);
  }
  */
    //bc.set(BoundaryCondition::DIRICHLET, 0.0, 0);
    //bc.set(BoundaryCondition::DIRICHLET, 0.0, 1);

  /*
  // u = 0 on the inflow boundary
  if ( bc.coord().x == 0.0 )
    bc.set(BoundaryCondition::NEUMANN, 0.0);
  else if ( bc.coord().x == 1.0 )
    bc.set(BoundaryCondition::DIRICHLET, 0.0);
  else if ( bc.coord().y == 0.0)
    bc.set(BoundaryCondition::NEUMANN, 0.0);
  else if ( bc.coord().y == 1.0 )
    bc.set(BoundaryCondition::NEUMANN, 0.0);
  else
    bc.set(BoundaryCondition::DIRICHLET, 1.0);
  */
}

int main(int argc, char **argv)
{
  //Mesh mesh("dolfin.xml.gz");
  //Mesh mesh("dolfin.xml.gz");
  //Mesh mesh("trimesh-16.xml.gz");
  //Mesh mesh("trimesh-2.xml.gz");
  //Mesh mesh("trimesh-4.xml.gz");
  //Mesh mesh("trimesh-64.xml.gz");
  //Mesh mesh("trimesh-32.xml.gz");

  //Mesh mesh("tetmesh-4.xml.gz");
  Mesh mesh("tetmesh-8.xml.gz");

  Problem elasticity("elasticity", mesh);

  elasticity.set("source", f);
  elasticity.set("diffusivity", a);
  elasticity.set("convection", b);
  elasticity.set("boundary condition", mybc);
  elasticity.set("final time", 2.0);
  elasticity.set("time step", 0.1);

  elasticity.solve();
  
  return 0;
}
