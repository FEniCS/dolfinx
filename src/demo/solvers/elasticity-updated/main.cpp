// Copyright (C) 2004-2005 Johan Jansson.
// Licensed under the GNU GPL Version 2.
//
// First added:  2004
// Last changed: 2005

#include <dolfin.h>

using namespace dolfin;

// Density
class Density : public Function
{
  real operator() (const Point& p) const
  {
    if(p.x < 0.01 && p.y > 0.0)
      return 1.0e3;
    else
      return 1.0e3;
  }
};

// Right-hand side
class Source : public Function
{
  real operator() (const Point& p, unsigned int i) const
  {
    int id = _cell->id();

    if(i == 1)
      return 0.0;
    else
      return 0.0;
  }
};

// Initial velocity
class InitialVelocity : public Function
{
  real operator() (const Point& p, unsigned int i) const
  {
    real result = 0.0;

    Point w, r, v, center;

    center.x = 1.0;
    center.y = 1.0;
    center.z = 0.0;

    r = p - center;

    w.x = 0.0;
    w.y = 0.0;
    w.z = 2.0;

    v = w.cross(r);

    if(i == 0)
    {
      result += v.x;
    }
    if(i == 1)
    {
      result += v.y;
    }


    return result;
  }
};

// Boundary condition
class MyBC : public BoundaryCondition
{
public:
  MyBC::MyBC() : BoundaryCondition(3)
  {
  }

//   const BoundaryValue operator() (const Point& p, int i)
//   {
//     BoundaryValue value;

//     if(p.x == 0.0 && p.y == 0.0)
//     {
//       value.set(0.0);
//     }    

//     return value;
//   }
};

int main(int argc, char **argv)
{
  dolfin_output("plain text");
//   dolfin_output("silent");

  //Mesh mesh("cow01.xml.gz");
  //Mesh mesh("cow07.xml.gz");
  //Mesh mesh("mymesh01.xml.gz");
//   Mesh mesh("minimal3.xml.gz");

  Mesh mesh("roterror01.xml.gz");
//   UnitCube mesh(7, 7, 7);

//   for (NodeIterator n(&mesh); !n.end();)
//   {
//     Node &cnode = *n;
//     ++n;

//     if(cnode.coord().y > 10.0)
//     {
//       mesh.remove(cnode);
//     }
//   }

//   mesh.init();

//   File outfile("unitcube-07.xml.gz");
//   outfile << mesh;
  
  Source f;
  Density rho;
  InitialVelocity v0;
  MyBC bc;

//   real T = 0.1;  // final time
//   real k = 0.001; // time step

  real T = 5.0;  // final time
  real k = 1.0e-3; // time step

//   real E = 1.0e4; // Young's modulus
//   real nu = 0.3; // Poisson's ratio
//   real nuv = 1.0e4; // viscosity

//   real E = 1.0e5; // Young's modulus
//   real nu = 0.3; // Poisson's ratio
//   real nuv = 1.0e3; // viscosity

  real E = 5.0e4; // Young's modulus
  real nu = 0.3; // Poisson's ratio
  real nuv = 1.0e2; // viscosity
  real nuplast = 0.0; // plastic viscosity

  ElasticityUpdatedSolver::solve(mesh, f, v0, rho, E, nu, nuv,
				 nuplast, bc, k, T);

  return 0;
}

