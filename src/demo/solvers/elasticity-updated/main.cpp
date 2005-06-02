// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin.h>

using namespace dolfin;

// Right-hand side
class Source : public Function
{
  real operator() (const Point& p, unsigned int i) const
  {
    if(i == 1)
      return -1.0;
    else
      return 0.0;
  }
};

// Initial displacement
class InitialVelocity : public Function
{
  real operator() (const Point& p, unsigned int i) const
  {
    //    if(i == 0)
    if(i == 0 && p.x > 0.5 )
    {
      return 1.0;
    }
    else if(i == 0 && p.x <= 0.5 && p.y > 0.5)
    {
      return -0.5;
    }
    else if(i == 0 && p.x <= 0.5 && p.y <= 0.5)
    {
      return -0.5;
    }
//     else if(i == 1 && p.x > 0.5 )
//     {
//       return 0.5;
//     }
    else
    {
      return 0.0;
    }
  }
};

// Boundary condition
class MyBC : public BoundaryCondition
{
public:
  MyBC::MyBC() : BoundaryCondition(3)
  {
  }

  const BoundaryValue operator() (const Point& p, int i)
  {
    BoundaryValue value;

    if(p.x == 0.0)
    {
      value.set(0.0);
    }    

    return value;
  }
};

int main(int argc, char **argv)
{
  dolfin_output("text");

  Mesh mesh("minimal2.xml.gz");
  //Mesh mesh("mymesh01.xml.gz");

//   File mymesh("mymesh.xml.gz");

   for(int i = 0; i < 2; i++)
   {
     mesh.refineUniformly();
   }

//   mymesh << mesh;

  Source f;
  InitialVelocity v0;
  MyBC bc;

//   real T = 0.1;  // final time
//   real k = 0.001; // time step

  real T = 5.0;  // final time
  real k = 0.01; // time step

  real E = 10.0; // Young's modulus
  real nu = 0.3; // Poisson's ratio

  ElasticityUpdatedSolver::solve(mesh, f, v0, E, nu, bc, k, T);

  return 0;
}
