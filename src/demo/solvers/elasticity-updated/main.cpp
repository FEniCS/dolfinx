// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin.h>

using namespace dolfin;

// Source term
real f(real x, real y, real z, real t, int i)
{
  real density;
  density = 1;

  real force = 0;


  /*
  if(i == 0 && t < 2.0)
    force += density * 30.0;
  */

  /*
  if(i == 1 && t < 3.0)
    force += -density * 2.0;
  */

  ///*
  if(i == 1)
    force += -density * 9.81 * 1.0;
  //*/

  /*
  if(i == 1 && x > 0.7 && y > 0.7)
  {
    force += 30;
  }

  if(i == 0 && x > 0.7 && y > 0.7)
  {
    force += 30;
  }
  */

  return force;
}

// Initial velocity
real v0(real x, real y, real z, real t, int i)
{
  real velocity = 0;

  if(i == 0 && y < 0.7)
  {
    velocity += 0.0;
  }

  /*
  if(i == 1 && x > 0.5)
  {
    velocity += 1.0;
  }

  if(i == 1 && x < -0.5)
  {
    velocity -= 1.0;
  }

  if(i == 0 && y > 0.5)
  {
    velocity -= 1.0;
  }

  if(i == 0 && y < -0.5)
  {
    velocity += 1.0;
  }
  */

  /*
  if(i == 1 && x > 2.0)
    velocity += 1.0;
  if(i == 1 && x < 1.0)
    velocity += 1.0;
  if(i == 1 && x >= 1.0 && x <= 2.0)
    velocity -= 1.0;
  */

  /*
  if(i == 1 && x >= 0.3)
    velocity += 3.0;

  if(i == 1 && x < 0.3 && x >= -0.3)
    velocity -= 3.0;

  if(i == 1 && x < -0.3)
    velocity += 3.0;
  */

  /*
  if(i == 1 && x >= 0.3)
    velocity += 2.5;

  if(i == 1 && x < 0.3 && x >= -0.3)
    velocity -= 3.5;

  if(i == 1 && x < -0.3)
    velocity += 2.5;
  */

  /*
  if(i == 1 && x >= 0.3)
    velocity -= 2.5;

  if(i == 1 && x < 0.3 && x >= -0.3)
    velocity += 3.5;

  if(i == 1 && x < -0.3)
    velocity -= 2.5;
  */


  /*
  if(i == 1 && x > 0.5)
    velocity += 2.0;
  if(i == 0 && y > 0.5)
    velocity += 1.0;
  */

  //if(i == 0)
  //velocity += 3.0;


  ///*

  /*
  if(i == 0)
    velocity += 3.0;
  */
  /*
  if(i == 2 && x > 0.5)
    velocity += 3.0;
  */
  /*
  if(i == 0 && x > 0.8)
    velocity += 2.0;
  if(i == 0 && x < -0.8)
    velocity -= 2.0;
  if(i == 0 && y > 0.8)
    velocity += 2.0;
  */

  //  if(i == 0 && x < 0.4)
  //    velocity -= 1.0;

  //  if(i == 0 && y < 0.4)
  //    velocity += 3.0;

  //*/

  return velocity;
}

// Boundary conditions
void mybc(BoundaryCondition& bc)
{
  ///*
  //if(fabs(bc.coord().x) < 0.1 && fabs(bc.coord().y) < 0.1)
  //if(fabs(bc.coord().x) < 0.1 && fabs(bc.coord().z) < 0.1 && fabs(bc.coord().y) < 0.7)
  //if(bc.coord().x == 0.0 && bc.coord().y == 0)
  //if(bc.coord().y == 1.0)
  //if(bc.coord().y > 0.7)
  //if(bc.coord().x == 0.0 && bc.coord().z == 0.0)
  if(bc.coord().x <= 0.0 && bc.coord().y >= 0.0)
  {
    bc.set(BoundaryCondition::DIRICHLET, 0.0, 0);
    bc.set(BoundaryCondition::DIRICHLET, 0.0, 1);
    bc.set(BoundaryCondition::DIRICHLET, 0.0, 2);
  }
  //*/
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

  dolfin_set("output", "plain text");

  //Mesh mesh("cow01.xml.gz");
  //Mesh mesh("cow02.xml.gz");
  //Mesh mesh("tetmesh-1c.xml.gz");
  //Mesh mesh("minimal2.xml.gz");
  Mesh mesh("tetmesh-1.xml.gz");
  //Mesh mesh("tetmesh-4.xml.gz");
  //Mesh mesh("tetmesh-8.xml.gz");

  //Mesh mesh("diamond-1.xml.gz");

  mesh.refineUniformly();
  //mesh.refineUniformly();

  Problem elasticity("elasticity-updated", mesh);
  //Problem elasticity("elasticity", mesh);

  elasticity.set("source", f);
  elasticity.set("initial velocity", v0);
  elasticity.set("boundary condition", mybc);
  elasticity.set("final time", 5.0);
  elasticity.set("time step", 0.001);

  elasticity.solve();
  
  return 0;
}
