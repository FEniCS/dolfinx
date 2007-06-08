// Copyright (C) 2005-2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2005
// Last changed: 2007-05-24

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
class DirichletBoundary : public SubDomain
{
  bool inside(const real* x, bool on_boundary) const
  {
    return on_boundary;
  }
};

// Right-hand side, 2D
class Source2D : public Function
{
public:

  Source2D(Mesh& mesh) : Function(mesh) {}

  real eval(const real* x) const
  {
    return 2.0*DOLFIN_PI*DOLFIN_PI*sin(DOLFIN_PI*x[0])*sin(DOLFIN_PI*x[1]);
  }

};

// Right-hand side, 3D
class Source3D : public Function
{
public:
  
  Source3D(Mesh& mesh) : Function(mesh) {}

  real eval(const real* x) const
  {
    return 3.0*DOLFIN_PI*DOLFIN_PI*sin(DOLFIN_PI*x[0])*sin(DOLFIN_PI*x[1])*sin(DOLFIN_PI*x[2]);
  }

};

// Solve equation and compute error, 2D
real solve2D(int q, int n)
{
  message("--------------------------------------------------");
  message("Solving Poisson's equation in 2D for q = %d, n = %d.", q, n);
  message("--------------------------------------------------");

  // Set up problem
  UnitSquare mesh(n, n);
  Source2D f(mesh);
  Function zero(mesh, 0.0);
  DirichletBoundary boundary;
  BoundaryCondition bc(zero, mesh, boundary);

  // Choose forms
  Form* a = 0;
  Form* L = 0;
  switch ( q )
  {
  case 1:
    a = new Poisson2D_1BilinearForm();
    L = new Poisson2D_1LinearForm(f);
    break;
  case 2:
    a = new Poisson2D_2BilinearForm();
    L = new Poisson2D_2LinearForm(f);
    break;
  case 3:
    a = new Poisson2D_3BilinearForm();
    L = new Poisson2D_3LinearForm(f);
    break;
  case 4:
    a = new Poisson2D_4BilinearForm();
    L = new Poisson2D_4LinearForm(f);
    break;
  case 5:
    a = new Poisson2D_5BilinearForm();
    L = new Poisson2D_5LinearForm(f);
    break;
  default:
    error("Forms not compiled for q = %d.", q);
  }    

  // Discretize equation
  Matrix A;
  Vector x, b;
  assemble(A, *a, mesh);
  assemble(b, *L, mesh);
  bc.apply(A, b, *a);

  // Solve the linear system
  KrylovSolver solver(gmres);
  solver.set("Krylov relative tolerance", 1e-14); 
  solver.solve(A, x, b);

  // Compute maximum norm of error
  real emax = 0.0;
  for (VertexIterator v(mesh); !v.end(); ++v)
  {
    const Point p = v->point();
    const real U = x(v->index());
    const real u = sin(DOLFIN_PI*p.x())*sin(DOLFIN_PI*p.y());
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
  message("--------------------------------------------------");
  message("Solving Poisson's equation in 3D for q = %d, n = %d.", q, n);
  message("--------------------------------------------------");

  // Set up problem
  UnitCube mesh(n, n, n);
  Source3D f(mesh);
  Function zero(mesh, 0.0);
  DirichletBoundary boundary;
  BoundaryCondition bc(zero, mesh, boundary);

  // Choose forms
  Form* a = 0;
  Form* L = 0;
  switch ( q )
  {
  case 1:
    a = new Poisson3D_1BilinearForm();
    L = new Poisson3D_1LinearForm(f);
    break;
  case 2:
    a = new Poisson3D_2BilinearForm();
    L = new Poisson3D_2LinearForm(f);
    break;
  case 3:
    a = new Poisson3D_3BilinearForm();
    L = new Poisson3D_3LinearForm(f);
    break;
  case 4:
    a = new Poisson3D_4BilinearForm();
    L = new Poisson3D_4LinearForm(f);
    break;
  case 5:
    a = new Poisson3D_5BilinearForm();
    L = new Poisson3D_5LinearForm(f);
    break;
  default:
    error("Forms not compiled for q = %d.", q);
  }    

  // Discretize equation
  Matrix A;
  Vector x, b;
  assemble(A, *a, mesh);
  assemble(b, *L, mesh);
  bc.apply(A, b, *a);

  // Solve the linear system
  KrylovSolver solver(gmres);
  solver.set("Krylov relative tolerance", 1e-14); 
  solver.solve(A, x, b);

  // Compute maximum norm of error
  real emax = 0.0;
  for (VertexIterator v(mesh); !v.end(); ++v)
  {
    const Point p = v->point();
    const real U = x(v->index());
    const real u = sin(DOLFIN_PI*p.x())*sin(DOLFIN_PI*p.y())*sin(DOLFIN_PI*p.z());
    const real e = std::abs(U - u);
    emax = std::max(emax, e);
  }

  delete a;
  delete L;

  return emax;
}

int main()
{
  const int qmax = 5;
  const int num_meshes = 3;
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
