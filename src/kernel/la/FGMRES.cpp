// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/FGMRES.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
FGMRES::FGMRES(const Matrix& A, real tol, unsigned int maxiter)
  : A(A), tol(tol), maxiter(maxiter)
{
  dolfin_error("Not implemented.");
}
//-----------------------------------------------------------------------------
FGMRES::~FGMRES()
{
  dolfin_error("Not implemented.");
}
//-----------------------------------------------------------------------------
void FGMRES::solve(Vector& x, const Vector& b)
{
  dolfin_error("Not implemented.");
}
//-----------------------------------------------------------------------------
void FGMRES::solve(const Matrix& A, Vector& x, const Vector& b,
		  real tol, unsigned int maxiter)
{
  // Create a FGMRES object
  FGMRES fgmres(A, tol, maxiter);

  // Solve linear system
  fgmres.solve(x, b);
}
//-----------------------------------------------------------------------------
