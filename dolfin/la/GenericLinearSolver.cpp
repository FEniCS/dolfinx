// Copyright (C) 2011 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2011-03-23
// Last changed: 2011-03-24

#include "GenericMatrix.h"
#include "GenericVector.h"
#include "GenericLinearSolver.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void GenericLinearSolver::check_dimensions(const GenericMatrix& A,
                                           GenericVector& x,
                                           const GenericVector& b) const
{
  // Check dimensions of A
  if (A.size(0) == 0 || A.size(1) == 0)
    error("Unable to solve linear system; matrix must have a nonzero number of rows and columns.");

  // Check dimensions of A vs b
  if (A.size(0) != b.size())
    error("Unable to solve linear system; matrix dimension (%d rows) does not match dimension of righ-hand side vector (%d).",
          A.size(0), b.size());

  // Check dimensions of A vs x
  if (x.size() > 0 && x.size() != A.size(1))
    error("Unable to solve linear system; matrix dimension (%d columns) does not match dimension of solution vector (%d).",
          A.size(1), x.size());

  // FIXME: We could implement a more thorough check of local/global
  // FIXME: dimensions for distributed matrices and vectors here.
}
//-----------------------------------------------------------------------------
