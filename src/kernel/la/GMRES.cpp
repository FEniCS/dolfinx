// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/GMRES.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
GMRES::GMRES(const Matrix& A, real tol, unsigned int maxiter)
  : A(A), tol(tol), maxiter(maxiter)
{
  dolfin_error("Not implemented.");
}
//-----------------------------------------------------------------------------
GMRES::~GMRES()
{
  dolfin_error("Not implemented.");
}
//-----------------------------------------------------------------------------
void GMRES::solve(Vector& x, const Vector& b)
{
  dolfin_error("Not implemented.");
}
//-----------------------------------------------------------------------------
void GMRES::solve(const Matrix& A, Vector& x, const Vector& b,
		  real tol, unsigned int maxiter)
{
  // Create a GMRES object
  GMRES gmres(A, tol, maxiter);

  // Solve linear system
  gmres.solve(x, b);
}
//-----------------------------------------------------------------------------
