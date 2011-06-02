// Copyright (C) 2011 Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
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
