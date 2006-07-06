// Copyright (C) 2006 Garth N. Wells
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-05-29
// Last changed: 2006-07-05

#ifndef __UBLAS_SPARSE_MATRIX_H
#define __UBLAS_SPARSE_MATRIX_H

#include <dolfin/dolfin_log.h>
#include <dolfin/uBlasMatrix.h>

namespace dolfin
{

  /// This class represents a uBlas sparse matrix of dimension M x N.

  class uBlasSparseMatrix : public uBlasMatrix<ublas_sparse_matrix>
  {
  public:
    
    /// Constructor
    uBlasSparseMatrix();

    /// Constructor
    uBlasSparseMatrix(const uint M, const uint N);

    /// Output
    friend LogStream& operator<< (LogStream& stream, const uBlasSparseMatrix& A);

  };
}

#endif
