// Copyright (C) 2006 Garth N. Wells
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-05-29
// Last changed: 

#ifndef __UBLAS_DENSE_MATRIX_H
#define __UBLAS_DENSE_MATRIX_H

#include <dolfin/uBlasMatrix.h>
#include <dolfin/ublas.h>

namespace dolfin
{
  class uBlasDenseMatrix : public uBlasMatrix<ublas_dense_matrix> 
  {

  public:
  
    /// Constructor
    uBlasDenseMatrix();

    /// Constructor
    uBlasDenseMatrix(const uint M, const uint N);  
    
    /// Output
    friend LogStream& operator<< (LogStream& stream, const uBlasDenseMatrix& A);

  };
}

#endif
