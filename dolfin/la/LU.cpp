// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-08-16
// Last changed: 2006-08-16

#include "LU.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void LU::solve(const GenericMatrix& A, GenericVector& x,
	       const GenericVector& b)
{
  LUSolver solver;
  solver.solve(A, x, b);
}
