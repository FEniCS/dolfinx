// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin.h>

using namespace dolfin;

// Right-hand side
class Source : public NewFunction
{
  real operator() (const Point& p, unsigned int i) const
  {
//     if(i == 0)
//     {
//       if(time() >= 0.0 && time() < 0.5 && p.x > 0.99)
//       {
// 	return -20.0;
//       }
//       else if(time() >= 0.0 && time() < 0.5 && p.x < 0.01)
//       {
// 	return 20.0;
//       }
//     }
//     else
//     {
//       return 0.0;
//     }
//     return 0.0;
    if(i == 1)
      return -8.0;
    else
      return 0.0;
  }
};

// Boundary condition
class MyBC : public NewBoundaryCondition
{
public:
  MyBC::MyBC() : NewBoundaryCondition(3)
  {
  }

  const BoundaryValue operator() (const Point& p, int i)
  {
    BoundaryValue value;

    if ( p.x == 0.0 )
    {
      value.set(0.0);
    }    

    return value;
  }
};

int main(int argc, char **argv)
{
  dolfin_output("plain text");

  Mesh mesh("tetmesh-4.xml.gz");

  Source f;
  MyBC bc;

  ElasticitySolver::solve(mesh, f, bc);

  return 0;
}
