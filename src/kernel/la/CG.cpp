// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/CG.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
CG::CG(const Matrix& A, real tol, unsigned int maxiter)
  : A(A), tol(tol), maxiter(maxiter)
{
  dolfin_error("Not implemented.");
}
//-----------------------------------------------------------------------------
CG::~CG()
{
  dolfin_error("Not implemented.");
}
//-----------------------------------------------------------------------------
void CG::solve(Vector& x, const Vector& b)
{
  dolfin_error("Not implemented.");
}
//-----------------------------------------------------------------------------
void CG::solve(const Matrix& A, Vector& x, const Vector& b,
		  real tol, unsigned int maxiter)
{
  // Create a CG object
  CG cg(A, tol, maxiter);

  // Solve linear system
  cg.solve(x, b);
}
//-----------------------------------------------------------------------------
