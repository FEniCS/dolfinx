// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin.h>

using namespace dolfin;

// Source term
real f(real x, real y, real z, real t, int i)
{
  if(i == 0)
  {
    if(x > 0.99)
    {
      return -10.0;
    }
    else if(x < 0.01)
    {
      return 10.0;
    }
  }

  return 0.0;
}

// Boundary conditions
void mybc(BoundaryCondition& bc)
{
  if(bc.coord().x == 0.5 && bc.coord().y == 0.5)
  {
    bc.set(BoundaryCondition::DIRICHLET, 0.0, 0);
    bc.set(BoundaryCondition::DIRICHLET, 0.0, 1);
    bc.set(BoundaryCondition::DIRICHLET, 0.0, 2);
  }
  if(bc.coord().x == 0.5 && bc.coord().y == 0.0)
  {
    bc.set(BoundaryCondition::DIRICHLET, 0.0, 0);
    bc.set(BoundaryCondition::DIRICHLET, 0.0, 1);
    bc.set(BoundaryCondition::DIRICHLET, 0.0, 2);
  }

  //bc.set(BoundaryCondition::DIRICHLET, 0.0, 0);
  
  /*
  if(bc.coord().x == 0.0)
  {
    bc.set(BoundaryCondition::DIRICHLET, 0.0, 0);
    bc.set(BoundaryCondition::DIRICHLET, 0.0, 1);
    bc.set(BoundaryCondition::DIRICHLET, 0.0, 2);
  }
  else if(bc.coord().y == 0.0)
  {
    bc.set(BoundaryCondition::DIRICHLET, 0.0, 0);
    bc.set(BoundaryCondition::DIRICHLET, 0.0, 1);
    bc.set(BoundaryCondition::DIRICHLET, 0.0, 2);
  }
  else if(bc.coord().z == 0.0)
  {
    bc.set(BoundaryCondition::DIRICHLET, 0.0, 0);
    bc.set(BoundaryCondition::DIRICHLET, 0.0, 1);
    bc.set(BoundaryCondition::DIRICHLET, 0.0, 2);
  }
  else if(bc.coord().x == 1.0)
  {
    bc.set(BoundaryCondition::DIRICHLET, 0.0, 0);
    bc.set(BoundaryCondition::DIRICHLET, 0.0, 1);
    bc.set(BoundaryCondition::DIRICHLET, 0.0, 2);
  }
  else if(bc.coord().y == 1.0)
  {
    bc.set(BoundaryCondition::DIRICHLET, 0.0, 0);
    bc.set(BoundaryCondition::DIRICHLET, 0.0, 1);
    bc.set(BoundaryCondition::DIRICHLET, 0.0, 2);
  }
  else if(bc.coord().z == 1.0)
  {
    bc.set(BoundaryCondition::DIRICHLET, 0.0, 0);
    bc.set(BoundaryCondition::DIRICHLET, 0.0, 1);
    bc.set(BoundaryCondition::DIRICHLET, 0.0, 2);
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
  Mesh mesh("tetmesh-4.xml.gz");
  Problem elasticitystationary("elasticity-stationary", mesh);

  mesh.refineUniformly();

  elasticitystationary.set("source", f);
  elasticitystationary.set("boundary condition", mybc);
  elasticitystationary.set("final time", 2.0);
  elasticitystationary.set("time step", 0.1);

  elasticitystationary.solve();
  
  return 0;
}
