// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __VIRTUAL_MATRIX_H
#define __VIRTUAL_MATRIX_H

#include <petsc/petscmat.h>
#include <dolfin/constants.h>
#include <dolfin/dolfin_log.h>

namespace dolfin
{

  class NewVector;
  
  /// This class represents a matrix-free matrix of dimension m x n.
  /// It is a simple wrapper for a PETSc shell matrix. The interface
  /// is intentionally simple. For advanced usage, access the PETSc
  /// Mat pointer using the function mat() and use the standard PETSc
  /// interface.
  ///
  /// The class VirtualMatrix enables the use of Krylov subspace
  /// methods for linear systems Ax = b, without having to explicitly
  /// store the matrix A. All that is needed is that the user-defined
  /// VirtualMatrix implements multiplication with vectors.

  class VirtualMatrix
  {
  public:

    /// Constructor
    VirtualMatrix();

    /// Constructor
    VirtualMatrix(uint m, uint n);

    /// Destructor
    virtual ~VirtualMatrix();

    /// Initialize matrix
    void init(uint m, uint n);

    /// Return number of rows (dim = 0) or columns (dim = 1) along dimension dim
    uint size(uint dim) const;

    /// Return PETSc Mat pointer
    Mat mat();

    /// Return PETSc Mat pointer
    const Mat mat() const;

    /// Compute product y = Ax
    virtual void mult(const NewVector& x, NewVector& y) const = 0;

    /// Output
    friend LogStream& operator<< (LogStream& stream, const VirtualMatrix& A);
    
  private:

    // PETSc Mat pointer
    Mat A;

  };

}

#endif
