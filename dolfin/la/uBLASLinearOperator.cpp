// Copyright (C) 2006 Anders Logg
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
// First added:  2006-07-07
// Last changed: 2012-08-21

#include "uBLASVector.h"
#include "uBLASSparseMatrix.h"
#include "uBLASLinearOperator.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void uBLASLinearOperator::solve(uBLASVector& x, const uBLASVector& b)
{
  // The linear system is solved by computing a dense copy of the matrix,
  // obtained through multiplication with unit vectors.

  // Check dimensions
  const uint M  = size(0);
  const uint N  = size(1);
  dolfin_assert(M == N);
  dolfin_assert(M == b.size());

  // Initialize temporary data if not already done
  if (!AA)
  {
    AA = new uBLASMatrix<ublas_dense_matrix>(M, N);
    ej = new uBLASVector(N);
    Aj = new uBLASVector(M);
  }
  else
  {
    AA->resize(M, N);
    ej->resize(N);
    Aj->resize(N);
  }

  // Get underlying uBLAS vectors
  ublas_vector& _ej = ej->vec();
  ublas_vector& _Aj = Aj->vec();
  ublas_dense_matrix& _AA = AA->mat();

  // Reset unit vector
  _ej *= 0.0;

  // Compute columns of matrix
  for (uint j = 0; j < N; j++)
  {
    (_ej)(j) = 1.0;

    // Compute product Aj = Aej
    mult(*ej, *Aj);

    // Set column of A
    column(_AA, j) = _Aj;

    (_ej)(j) = 0.0;
  }

  // Solve linear system
  warning("UmfpackLUSolver no longer solves dense matrices. This function will be removed and probably added to uBLASKrylovSolver.");
  warning("The uBLASKrylovSolver LU solver has been modified and has not yet been well tested. Please verify your results.");
 (*AA).solve(x, b);
}
//-----------------------------------------------------------------------------
std::string uBLASLinearOperator::str(bool verbose) const
{
  std::stringstream s;

  if (verbose)
  {
    warning("Verbose output for uBLASLinearOperator not implemented.");
  }
  else
  {
    s << "<uBLASLinearOperator of size " << size(0) << " x " << size(1) << ">";
  }

  return s.str();
}
//-----------------------------------------------------------------------------
