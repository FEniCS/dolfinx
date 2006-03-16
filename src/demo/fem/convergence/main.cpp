// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005
// Last changed: 2005-12-28

#include <stdio.h>
#include <dolfin.h>
#include "Poisson2D_1.h"
#include "Poisson2D_2.h"
#include "Poisson2D_3.h"
#include "Poisson2D_4.h"
#include "Poisson2D_5.h"
#include "Poisson3D_1.h"
#include "Poisson3D_2.h"
#include "Poisson3D_3.h"
#include "Poisson3D_4.h"
#include "Poisson3D_5.h"

using namespace dolfin;

// Boundary condition
class BC : public BoundaryCondition
{
  void eval(BoundaryValue& value, const Point& p, unsigned int i)
  {
    value = 0.0;
  }
};

// Right-hand side, 2D
class Source2D : public Function
{
  real eval(const Point& p, unsigned int i)
  {
    return 2.0*DOLFIN_PI*DOLFIN_PI*sin(DOLFIN_PI*p.x)*sin(DOLFIN_PI*p.y);
  }
};

// Right-hand side, 3D
class Source3D : public Function
{
  real eval(const Point& p, unsigned int i)
  {
    return 3.0*DOLFIN_PI*DOLFIN_PI*sin(DOLFIN_PI*p.x)*sin(DOLFIN_PI*p.y)*sin(DOLFIN_PI*p.z);
  }
};

// Solve equation and compute error, 2D
real solve2D(int q, int n)
{
  dolfin_info("--------------------------------------------------");
  dolfin_info("Solving Poisson's equation in 2D for q = %d, n = %d.", q, n);
  dolfin_info("--------------------------------------------------");

  // Set up problem
  UnitSquare mesh(n, n);
  Source2D f;
  BC bc;

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
  case 3:
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
    break;
  default:
    dolfin_error1("Forms not compiled for q = %d.", q);
  }    

  //FEM::disp(mesh, a->test());
  
  // Discretize equation
  Matrix A;
  Vector x, b;
  FEM::assemble(*a, *L, A, b, mesh, bc);

  // Solve the linear system
  GMRES solver;
  solver.set("Krylov relative tolerance", 1e-14); 
  solver.solve(A, x, b);

  // Compute maximum norm of error
  real emax = 0.0;
  for (VertexIterator n(mesh); !n.end(); ++n)
  {
    const Point& p = n->coord();
    const real U = x(n->id());
    const real u = sin(DOLFIN_PI*p.x)*sin(DOLFIN_PI*p.y);
    const real e = std::abs(U - u);
    emax = std::max(emax, e);
  }

  delete a;
  delete L;

  return emax;
}

// Solve equation and compute error, 3D
real solve3D(int q, int n)
{
  dolfin_info("--------------------------------------------------");
  dolfin_info("Solving Poisson's equation in 3D for q = %d, n = %d.", q, n);
  dolfin_info("--------------------------------------------------");

  // Set up problem
  UnitCube mesh(n, n, n);
  Source3D f;
  BC bc;

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
  case 3:
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
    break;
  default:
    dolfin_error1("Forms not compiled for q = %d.", q);
  }    

  //FEM::disp(mesh, a->test());
  
  // Discretize equation
  Matrix A;
  Vector x, b;
  FEM::assemble(*a, *L, A, b, mesh, bc);

  cout << "Maximum number of nonzeros: " << A.nzmax() << endl;
  
  // Solve the linear system
  GMRES solver;
  solver.set("Krylov relative tolerance", 1e-14); 
  solver.solve(A, x, b);

  // Compute maximum norm of error
  real emax = 0.0;
  for (VertexIterator n(mesh); !n.end(); ++n)
  {
    const Point& p = n->coord();
    const real U = x(n->id());
    const real u = sin(DOLFIN_PI*p.x)*sin(DOLFIN_PI*p.y)*sin(DOLFIN_PI*p.z);
    const real e = std::abs(U - u);
    emax = std::max(emax, e);
  }

  delete a;
  delete L;

  return emax;
}

int main()
{
  int qmax = 5;
  int num_meshes = 3;
  real e2D[qmax][num_meshes];
  real e3D[qmax][num_meshes];

  // Compute errors in 2D
  for (int q = 1; q <= qmax; q++)
  {
    int n = 2;
    for (int i = 0; i < num_meshes; i++)
    {
      e2D[q - 1][i] = solve2D(q, n);
      n *= 2;
    }
  }

  // Compute errors in 3D
  for (int q = 1; q <= qmax; q++)
  {
    int n = 2;
    for (int i = 0; i < num_meshes; i++)
    {
      e3D[q - 1][i] = solve3D(q, n);
      n *= 2;
    }
  }

  // Write errors in 2D
  printf("\nMaximum norm error in 2D:\n");
  printf("-------------------------\n");
  for (int q = 1; q <= qmax; q++)
  {
    printf("q = %d:", q);
    for (int i = 0; i < num_meshes; i++)
      printf(" %.3e", e2D[q - 1][i]);
    printf("\n");
  }

  // Write errors in 3D
  printf("\nMaximum norm error in 3D:\n");
  printf("-------------------------\n");
  for (int q = 1; q <= qmax; q++)
  {
    printf("q = %d:", q);
    for (int i = 0; i < num_meshes; i++)
      printf(" %.3e", e3D[q - 1][i]);
    printf("\n");
  }
}
