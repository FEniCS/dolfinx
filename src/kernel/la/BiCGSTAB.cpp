// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/BiCGSTAB.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
BiCGSTAB::BiCGSTAB(const Matrix& A, real tol, unsigned int maxiter)
  : A(A), tol(tol), maxiter(maxiter)
{
  dolfin_error("Not implemented.");
}
//-----------------------------------------------------------------------------
BiCGSTAB::~BiCGSTAB()
{
  dolfin_error("Not implemented.");
}
//-----------------------------------------------------------------------------
void BiCGSTAB::solve(Vector& x, const Vector& b)
{
  dolfin_error("Not implemented.");
}
//-----------------------------------------------------------------------------
void BiCGSTAB::solve(const Matrix& A, Vector& x, const Vector& b,
		  real tol, unsigned int maxiter)
{
  // Create a BiCGSTAB object
  BiCGSTAB bicgstab(A, tol, maxiter);

  // Solve linear system
  bicgstab.solve(x, b);
}
//-----------------------------------------------------------------------------
