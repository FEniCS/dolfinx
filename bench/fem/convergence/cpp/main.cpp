// Copyright (C) 2005-2010 Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.

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

using namespace dolfin;

// Boundary condition
class DirichletBoundary : public SubDomain
{
public:

  DirichletBoundary() {}

  bool inside(const Array<double>& x, bool on_boundary) const
  { return on_boundary; }
};

// Right-hand side, 2D
class Source2D : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = 2.0*DOLFIN_PI*DOLFIN_PI*sin(DOLFIN_PI*x[0])*sin(DOLFIN_PI*x[1]);
  }
};

// Right-hand side, 3D
class Source3D : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = 3.0*DOLFIN_PI*DOLFIN_PI*sin(DOLFIN_PI*x[0])*sin(DOLFIN_PI*x[1])*sin(DOLFIN_PI*x[2]);
  }
};

// Solve equation and compute error, 2D
double solve2D(int q, int n)
{
  printf("Solving Poisson's equation in 2D for q = %d, n = %d.\n", q, n);

  // Set up problem
  auto mesh = std::make_shared<UnitSquareMesh>(n, n);
  auto f = std::make_shared<Source2D>();
  auto zero = std::make_shared<const Constant>(0.0);

  // Choose forms
  Form* a = 0;
  Form* L = 0;
  std::shared_ptr<const FunctionSpace> V;
  switch (q)
  {
  case 1:
    V = std::make_shared<Poisson2D_1::FunctionSpace>(mesh);
    a = new Poisson2D_1::BilinearForm(V, V);
    L = new Poisson2D_1::LinearForm(V, f);
    break;
  case 2:
    V = std::make_shared<Poisson2D_2::FunctionSpace>(mesh);
    a = new Poisson2D_2::BilinearForm(V, V);
    L = new Poisson2D_2::LinearForm(V, f);
    break;
  case 3:
    V = std::make_shared<Poisson2D_3::FunctionSpace>(mesh);
    a = new Poisson2D_3::BilinearForm(V, V);
    L = new Poisson2D_3::LinearForm(V, f);
    break;
  case 4:
    V = std::make_shared<Poisson2D_4::FunctionSpace>(mesh);
    a = new Poisson2D_4::BilinearForm(V, V);
    L = new Poisson2D_4::LinearForm(V, f);
    break;
  case 5:
    V = std::make_shared<Poisson2D_5::FunctionSpace>(mesh);
    a = new Poisson2D_5::BilinearForm(V, V);
    L = new Poisson2D_5::LinearForm(V, f);
    break;
  default:
    error("Forms not compiled for q = %d.", q);
  }

  // Set up boundary conditions
  auto boundary = std::make_shared<const DirichletBoundary>();
  DirichletBC bc(V, zero, boundary);

  // Discretize equation
  Matrix A;
  Vector x, b;
  assemble(A, *a);
  assemble(b, *L);
  bc.apply(A, b);

  // Solve the linear system
  KrylovSolver solver("gmres");
  solver.parameters["relative_tolerance"] = 1e-14;
  solver.solve(A, x, b);

  // Compute maximum norm of error
  double emax = 0.0;
  std::vector<double> U;
  x.get_local(U);
  for (VertexIterator v(*mesh); !v.end(); ++v)
  {
    const Point p = v->point();
    const double u = sin(DOLFIN_PI*p.x())*sin(DOLFIN_PI*p.y());
    const double e = std::abs(U[v->index()] - u);
    emax = std::max(emax, e);
  }

  delete a;
  delete L;

  return emax;
}

// Solve equation and compute error, 3D
double solve3D(int q, int n)
{
  printf("Solving Poisson's equation in 3D for q = %d, n = %d.\n", q, n);

  // Set up problem
  auto mesh = std::make_shared<UnitCubeMesh>(n, n, n);
  auto f = std::make_shared<Source3D>();
  auto zero = std::make_shared<const Constant>(0.0);

  // Choose forms
  Form* a = 0;
  Form* L = 0;
  std::shared_ptr<const FunctionSpace> V;
  switch (q)
  {
  case 1:
    V = std::make_shared<Poisson3D_1::FunctionSpace>(mesh);
    a = new Poisson3D_1::BilinearForm(V, V);
    L = new Poisson3D_1::LinearForm(V, f);
    break;
  case 2:
    V = std::make_shared<Poisson3D_2::FunctionSpace>(mesh);
    a = new Poisson3D_2::BilinearForm(V, V);
    L = new Poisson3D_2::LinearForm(V, f);
    break;
  case 3:
    V = std::make_shared<Poisson3D_3::FunctionSpace>(mesh);
    a = new Poisson3D_3::BilinearForm(V, V);
    L = new Poisson3D_3::LinearForm(V, f);
    break;
  case 4:
    V = std::make_shared<Poisson3D_4::FunctionSpace>(mesh);
    a = new Poisson3D_4::BilinearForm(V, V);
    L = new Poisson3D_4::LinearForm(V, f);
    break;
  default:
    error("Forms not compiled for q = %d.", q);
  }

  // Set up boundary conditions
  auto boundary = std::make_shared<const DirichletBoundary>();
  DirichletBC bc(V, zero, boundary);

  // Discretize equation
  Matrix A;
  Vector x, b;
  assemble(A, *a);
  assemble(b, *L);
  bc.apply(A, b);

  // Solve the linear system
  KrylovSolver solver("gmres");
  solver.parameters["relative_tolerance"] = 1e-14;
  solver.solve(A, x, b);

  // Compute maximum norm of error
  double emax = 0.0;
  std::vector<double> U;
  x.get_local(U);
  for (VertexIterator v(*mesh); !v.end(); ++v)
  {
    const Point p = v->point();
    const double u = sin(DOLFIN_PI*p.x())*sin(DOLFIN_PI*p.y())*sin(DOLFIN_PI*p.z());
    const double e = std::abs(U[v->index()] - u);
    emax = std::max(emax, e);
  }

  delete a;
  delete L;

  return emax;
}

int main()
{
  info("Runtime of convergence benchmark");

  set_log_active(false);

  const int qmax_2D = 5;
  const int qmax_3D = 4;
  const int num_meshes = 3;
  std::vector<std::vector<double>> e2D(qmax_2D,
                                       std::vector<double>(num_meshes));
  std::vector<std::vector<double>> e3D(qmax_3D,
                                       std::vector<double>(num_meshes));

  // Compute errors in 2D
  for (int q = 1; q <= qmax_2D; q++)
  {
    int n = 2;
    for (int i = 0; i < num_meshes; i++)
    {
      e2D[q - 1][i] = solve2D(q, n);
      n *= 2;
    }
  }

  // Compute errors in 3D
  for (int q = 1; q <= qmax_3D; q++)
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
  for (int q = 1; q <= qmax_2D; q++)
  {
    printf("q = %d:", q);
    for (int i = 0; i < num_meshes; i++)
      printf(" %.3e", e2D[q - 1][i]);
    printf("\n");
  }

  // Write errors in 3D
  printf("\nMaximum norm error in 3D:\n");
  printf("-------------------------\n");
  for (int q = 1; q <= qmax_3D; q++)
  {
    printf("q = %d:", q);
    for (int i = 0; i < num_meshes; i++)
      printf(" %.3e", e3D[q - 1][i]);
    printf("\n");
  }
}
