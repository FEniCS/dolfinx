// Copyright (C) 2007-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-04-30
// Last changed: 2008-01-10

#include <dolfin/LUSolver.h>
#include <dolfin/Matrix.h>
#include <dolfin/Vector.h>
#include <dolfin/solve.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
void dolfin::solve(const Matrix& A, Vector& x, const Vector& b)
{
  LUSolver solver;
  solver.solve(A, x, b);
}
//-----------------------------------------------------------------------------  
real dolfin::residual(const Matrix& A, const Vector& x, const Vector& b)
{
  Vector y;
  A.mat().mult(x.vec(), y.vec());
  y.vec() -= b.vec();
  return y.vec().norm();
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
