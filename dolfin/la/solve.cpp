// Copyright (C) 2007-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Ola Skavhaug 2008.
//
// First added:  2007-04-30
// Last changed: 2008-08-19

#include <dolfin/common/Timer.h>
#include "LinearSolver.h"
#include "GenericMatrix.h"
#include "GenericVector.h"
#include "LinearAlgebraFactory.h"
#include "solve.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void dolfin::solve(const GenericMatrix& A, GenericVector& x, const GenericVector& b,
                   SolverType solver_type, PreconditionerType pc_type)
{
  Timer timer("Solving linear system");
  LinearSolver solver(solver_type, pc_type);
  solver.solve(A, x, b);
}
//-----------------------------------------------------------------------------  
double dolfin::residual(const GenericMatrix& A, const GenericVector& x, const GenericVector& b)
{
  GenericVector* y = A.factory().create_vector();
  A.mult(x, *y);
  *y -= b;
  const double norm = y->norm(l2);
  delete y;
  return norm;
}
//-----------------------------------------------------------------------------
double dolfin::normalize(GenericVector& x, NormalizationType normalization_type)
{
  switch (normalization_type)
  {
  case normalize_l2norm:
    {
      const double c = x.norm(l2);
      x /= c;
      return c;
    }
    break;
  case normalize_average:
    {
      GenericVector* y = x.factory().create_vector();
      y->resize(x.size());
      (*y) = 1.0 / static_cast<double>(x.size());
      const double c = x.inner(*y);
      (*y) = c;
      x -= (*y);
      delete y;
      return c;
    }
    break;
  default:
    error("Unknown normalization type.");
  }

  return 0.0;
}
//-----------------------------------------------------------------------------
/*
void dolfin::solve(const PETScKrylovMatrix& A,
                   PETScVector& x,
                   const PETScVector& b)
{
  PETScLUSolver solver;
  solver.solve(A, x, b);
}
*/
//-----------------------------------------------------------------------------
