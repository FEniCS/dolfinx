// Copyright (C) 2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-12-15
// Last changed: 2009-12-15
//
// This file defines C++ versions of some LAPACK routines used in
// DOLFIN.

#ifndef __LAPACK_H
#define __LAPACK_H

extern "C"
{

  // Declaration of Fortran implementation of DGELS
  void dgels_(char* trans, int* m, int* n, int* nrhs,
              double* a, int* lda, double* b, int* ldb,
              double* work, int* lwork, int* info);

}

namespace dolfin
{

  // LAPACK DGELS routine (solve least squares using QR or LQ factorization)
  void dgels(char* trans, int* m, int* n, int* nrhs,
             double* a, int* lda, double* b, int* ldb,
             double* work, int* lwork, int* info)
  {  dgels_(trans, m, n, nrhs, a, lda, b, ldb, work, lwork, info); }

}

#endif
