// Copyright (C) 2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-12-14
// Last changed: 2010-05-08

#include <dolfin/log/log.h>
#include "lapack.h"
#include "LAPACKMatrix.h"
#include "LAPACKVector.h"
#include "LAPACKSolvers.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void LAPACKSolvers::solve_least_squares(const LAPACKMatrix& A,
                                        LAPACKVector& b)
{
  // Check dimensions
  assert(A.size(0) == b.size());

  // Prepare arguments for DGELS
  char trans = 'N';
  int m = A.size(0);
  int n = A.size(1);
  int nrhs = 1;
  int lda = m;
  int ldb = m;
  int lwork = std::max(1, std::min(m, n) + std::max(std::min(m, n), nrhs));
  int status = 0;
  double* work = new double[m*lwork];

  // Call DGELSS
  info(TRACE, "Solving least squares system of size %d x %d using DGELS.", m, n);
  dgels(&trans, &m, &n, &nrhs, A.values, &lda, b.values, &ldb, work, &lwork, &status);

  // Check output status
  if (status < 0)
    error("Illegal value for parameter number %d in call to DGELS.", status);
  else if (status > 0)
    error("Unable to solve least squares problem, matrix does not have full rank.");

  // Clean up
  delete [] work;
}
//-----------------------------------------------------------------------------
