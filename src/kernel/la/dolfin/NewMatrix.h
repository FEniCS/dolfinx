// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __NEW_MATRIX_H
#define __NEW_MATRIX_H

#include <petsc/petscmat.h>
#include <dolfin/constants.h>
#include <dolfin/dolfin_log.h>

namespace dolfin
{
  
  /// This class represents a matrix of dimension m x n. It is a
  /// simple wrapper for a PETSc matrix (Mat). The interface is
  /// intentionally simple. For advanced usage, access the PETSc Mat
  /// pointer using the function mat() and use the standard PETSc
  /// interface.

  class NewMatrix
  {
  public:

    /// Constructor
    NewMatrix();

    /// Constructor
    NewMatrix(int m, int n);

    /// Destructor
    ~NewMatrix();

    /// Initialize matrix
    void init(int m, int n);

    /// Return number of rows (dim = 0) or columns (dim = 1) along dimension dim
    int size(int dim) const;

    /// Set all entries to zero
    NewMatrix& operator= (real zero);

    /// Add block of values
    void add(const real block[], const int rows[], int m, const int cols[], int n);

    /// Apply changes to matrix
    void apply();

    /// Return PETSc Mat pointer
    Mat mat();

    /// Return PETSc Mat pointer
    const Mat mat() const;

    /// Display matrix
    void show() const;

    /// Condensed output
    friend LogStream& operator<< (LogStream& stream, const NewMatrix& A);
    
  private:

    // PETSc Mat pointer
    Mat A;

  };

}

#endif
