// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <stdio.h>
#include <dolfin.h>
#include "Poisson2D_1.h"
#include "Poisson2D_2.h"
//#include "Poisson2D_3.h"
//#include "Poisson2D_4.h"
//#include "Poisson2D_5.h"
#include "Poisson3D_1.h"
#include "Poisson3D_2.h"
//#include "Poisson3D_3.h"
//#include "Poisson3D_4.h"
//#include "Poisson3D_5.h"

using namespace dolfin;

// Boundary condition: homogeneous Dirichlet
class MyBC : public BoundaryCondition
{
  const BoundaryValue operator() (const Point& p)
  {
    BoundaryValue value;
    value.set(0.0);
    return value;
  }
};

// Right-hand side, 2D
class MyFunction2D : public Function
{
  real operator() (const Point& p) const
  {
    return 2.0*DOLFIN_PI*DOLFIN_PI*sin(DOLFIN_PI*p.x)*sin(DOLFIN_PI*p.y);
  }
};

// Right-hand side, 3D
class MyFunction3D : public Function
{
  real operator() (const Point& p) const
  {
    return 3.0*DOLFIN_PI*DOLFIN_PI*sin(DOLFIN_PI*p.x)*sin(DOLFIN_PI*p.y)*sin(DOLFIN_PI*p.z);
  }
};

// Solve equation and compute error, 2D
real solve2D(unsigned int q, unsigned int n)
{
  dolfin_info("----------------------------------------------------");
  dolfin_info("--- Solving Poisson's equation in 2D for q = %d, n = %d. ---", q, n);
  dolfin_info("----------------------------------------------------");

  // Set up problem
  UnitSquare mesh(n, n);
  MyFunction2D f;
  MyBC bc;

  // Choose forms
  BilinearForm* a;
  LinearForm* L;
  switch ( q )
  {
  case 1:
    a = new Poisson2D_1::BilinearForm();
    L = new Poisson2D_1::LinearForm(f);
    break;
  case 2:
    a = new Poisson2D_2::BilinearForm();
    L = new Poisson2D_2::LinearForm(f);
    break;
    /* case 3:
    a = new Poisson2D_3::BilinearForm();
    L = new Poisson2D_3::LinearForm(f);
    break;
  case 4:
    a = new Poisson2D_4::BilinearForm();
    L = new Poisson2D_4::LinearForm(f);
    break;
  case 5:
    a = new Poisson2D_5::BilinearForm();
    L = new Poisson2D_5::LinearForm(f);
    break; */
  default:
    dolfin_error1("Forms not compiled for q = %d.", q);
  }    
  
  // Discretize equation
  Matrix A;
  Vector x, b;
  FEM::assemble(*a, *L, A, b, mesh, bc);

  // Solve the linear system
  GMRES solver;
  solver.solve(A, x, b);

  // Compute maximum norm of error
  real emax = 0.0;
  for (NodeIterator n(mesh); !n.end(); ++n)
  {
    const Point& p = n->coord();
    const real u = sin(DOLFIN_PI*p.x)*sin(DOLFIN_PI*p.y);
    emax = std::max(emax, std::abs(x(n->id()) - u));
  }

  delete a;
  delete L;

  return emax;
}

// Solve equation and compute error, 3D
real solve3D(unsigned int q, unsigned int n)
{
  dolfin_info("----------------------------------------------------");
  dolfin_info("Solving Poisson's equation in 3D for q = %d, n = %d.", q, n);
  dolfin_info("----------------------------------------------------");
  
  // Set up problem
  UnitCube mesh(n, n, n);
  MyFunction3D f;
  MyBC bc;

  // Choose forms
  BilinearForm* a;
  LinearForm* L;
  switch ( q )
  {
  case 1:
    a = new Poisson3D_1::BilinearForm();
    L = new Poisson3D_1::LinearForm(f);
    break;
  case 2:
    a = new Poisson3D_2::BilinearForm();
    L = new Poisson3D_2::LinearForm(f);
    break;
    /* case 3:
    a = new Poisson3D_3::BilinearForm();
    L = new Poisson3D_3::LinearForm(f);
    break;
  case 4:
    a = new Poisson3D_4::BilinearForm();
    L = new Poisson3D_4::LinearForm(f);
    break;
  case 5:
    a = new Poisson3D_5::BilinearForm();
    L = new Poisson3D_5::LinearForm(f);
    break; */
  default:
    dolfin_error1("Forms not compiled for q = %d.", q);
  }    
  
  // Discretize equation
  Matrix A;
  Vector x, b;
  FEM::assemble(*a, *L, A, b, mesh, bc);

  // Solve the linear system
  GMRES solver;
  solver.solve(A, x, b);

  // Compute maximum norm of error
  real emax = 0.0;
  for (NodeIterator n(mesh); !n.end(); ++n)
  {
    const Point& p = n->coord();
    const real u = sin(DOLFIN_PI*p.x)*sin(DOLFIN_PI*p.y);
    emax = std::max(emax, std::abs(x(n->id()) - u));
  }

  delete a;
  delete L;

  return emax;
}

int main()
{
  unsigned int qmax = 2;
  unsigned int num_meshes = 5;
  real e2D[qmax][num_meshes];
  real e3D[qmax][num_meshes];

  // Compute errors in 2D
  for (unsigned int q = 1; q <= qmax; q++)
  {
    int n = 2;
    for (unsigned int i = 0; i < num_meshes; i++)
    {
      e2D[q - 1][i] = solve2D(q, n);
      n *= 2;
    }
  }

  // Compute errors in 3D
  for (unsigned int q = 1; q <= qmax; q++)
  {
    int n = 2;
    for (unsigned int i = 0; i < num_meshes; i++)
    {
      e3D[q - 1][i] = solve3D(q, n);
      n *= 2;
    }
  }

  // Write errors in 2D
  printf("\n");
  for (unsigned int q = 1; q <= qmax; q++)
  {
    printf("2D, q = %d:", q);
    for (unsigned int i = 0; i < num_meshes; i++)
      printf(" %.3e", e2D[q - 1][i]);
    printf("\n");
  }

  // Write errors in 3D
  printf("\n");
  for (unsigned int q = 1; q <= qmax; q++)
  {
    printf("3D, q = %d:", q);
    for (unsigned int i = 0; i < num_meshes; i++)
      printf(" %.3e", e3D[q - 1][i]);
    printf("\n");
  }
}
