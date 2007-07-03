// Copyright (C) 2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-04-30
// Last changed: 2007-04-30

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


