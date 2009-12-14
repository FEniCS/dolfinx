// Copyright (C) 2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-12-14
// Last changed: 2009-12-14

// FIXME: We are currently getting dgelss from CGAL.
// FIXME: Are there better options?

#ifdef HAS_CGAL
#include <CGAL/assertions.h>
#include <CGAL/Lapack/Linear_algebra_lapack.h>
#endif

#include <dolfin/log/log.h>
#include "LAPACKMatrix.h"
#include "LAPACKVector.h"
#include "LAPACKSolvers.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void LAPACKSolvers::solve_least_squares(const LAPACKMatrix& A,
                                        LAPACKVector& b)
{
#ifdef HAS_CGAL

  // Check dimensions
  assert(A.size(0) == b.size());

  // Prepare arguments fro DGELSS
  int m = A.size(0);
  int n = A.size(1);
  int nrhs = 1;
  int lda = m;
  int ldb = m;
  int lwork = 5;
  int rank = 0;
  int status = 0;
  double rcond = -1;
  double* s = new double[n];
  double* work = new double[m*lwork];

  // Call DGELSS
  info("Solving least squares system of size %d x %d using DGELSS.", m, n);
  CGAL::LAPACK::dgelss(&m, &n, &nrhs,
                       A.values, &lda, b.values, &ldb,
                       s, &rcond, &rank,
                       work, &lwork,
                       &status);

  // Check output status
  if (status < 0)
    error("Illegal value for parameter number %d in call to DGELSS.", status);
  else if (status > 0)
    error("Least squares solvers (SVD in DGELSS) did not converge.");

  // Report condition number
  info("Condition number is %g.", s[0] / s[n - 1]);

  // Clean up
  delete [] s;
  delete [] work;

#else

  error("Strangely enough, least squares solver is only available when DOLFIN is compiled with CGAL.");

#endif
}
//-----------------------------------------------------------------------------
