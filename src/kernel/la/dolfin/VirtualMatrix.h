// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-01-17
// Last changed: 2005

#ifndef __VIRTUAL_MATRIX_H
#define __VIRTUAL_MATRIX_H

#include <petscmat.h>
#include <dolfin/constants.h>
#include <dolfin/dolfin_log.h>

namespace dolfin
{

  class Vector;
  
  /// This class represents a matrix-free matrix of dimension m x m.
  /// It is a simple wrapper for a PETSc shell matrix. The interface
  /// is intentionally simple. For advanced usage, access the PETSc
  /// Mat pointer using the function mat() and use the standard PETSc
  /// interface.
  ///
  /// The class VirtualMatrix enables the use of Krylov subspace
  /// methods for linear systems Ax = b, without having to explicitly
  /// store the matrix A. All that is needed is that the user-defined
  /// VirtualMatrix implements multiplication with vectors. Note that
  /// the multiplication operator needs to be defined in terms of
  /// PETSc data structures (Vec), since it will be called from PETSc.

  class VirtualMatrix
  {
  public:

    /// Constructor
    VirtualMatrix();

    /// Create a virtual matrix matching the given vectors
    VirtualMatrix(const Vector& x, const Vector& y);

    /// Destructor
    virtual ~VirtualMatrix();

    /// Initialize virtual matrix matching the given vectors
    void init(const Vector& x, const Vector& y);

    /// Return number of rows (dim = 0) or columns (dim = 1) along dimension dim
    uint size(uint dim) const;

    /// Return PETSc Mat pointer
    Mat mat();

    /// Return PETSc Mat pointer
    const Mat mat() const;

    /// Compute product y = Ax
    virtual void mult(const Vector& x, Vector& y) const = 0;

    /// Display matrix (sparse output is default)
    void disp(bool sparse = true, int precision = 2) const;

    /// Output
    friend LogStream& operator<< (LogStream& stream, const VirtualMatrix& A);

  private:

    // PETSc Mat pointer
    Mat A;

  };

}

#endif
