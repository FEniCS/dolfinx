// Copyright (C) 2007-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Ola Skavhaug 2008.
//
// First added:  2007-04-30
// Last changed: 2008-04-11

#include "LUSolver.h"
#include "GenericMatrix.h"
#include "GenericVector.h"
#include "solve.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void dolfin::solve(const GenericMatrix& A, GenericVector& x, const GenericVector& b)
{
  LUSolver solver;
  solver.solve(A, x, b);
}
//-----------------------------------------------------------------------------  
real dolfin::residual(const GenericMatrix& A, const GenericVector& x, const GenericVector& b)
{
  DefaultVector* y = dynamic_cast<DefaultVector*>(A.factory().createVector());
  A.mult(x, *y);
  *y -= b;
  const real norm = y->norm();
  delete y;
  return norm;
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
