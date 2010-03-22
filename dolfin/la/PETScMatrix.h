// Copyright (C) 2004-2009 Johan Hoffman, Johan Jansson and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Andy R. Terrel, 2005.
// Modified by Garth N. Wells, 2006-2009.
// Modified by Kent-Andre Mardal, 2008.
// Modified by Ola Skavhaug, 2008.
//
// First added:  2004-01-01
// Last changed: 2009-12-14

#ifndef __PETSC_MATRIX_H
#define __PETSC_MATRIX_H

#ifdef HAS_PETSC

#include <map>
#include <string>
#include <boost/shared_ptr.hpp>
#include <petscmat.h>

#include "PETScObject.h"
#include "GenericMatrix.h"

namespace dolfin
{

  class PETScVector;

  /// This class provides a simple matrix class based on PETSc.
  /// It is a wrapper for a PETSc matrix pointer (Mat)
  /// implementing the GenericMatrix interface.
  ///
  /// The interface is intentionally simple. For advanced usage,
  /// access the PETSc Mat pointer using the function mat() and
  /// use the standard PETSc interface.
  ///
  /// std::string type is the type of PETSc matrix. Options are "default",
  /// "spooles", "superlu" and "umfpack". Note: Setting the matrix type may
  /// not be necessary in the future with PETSc version 3.

  class PETScMatrix : public GenericMatrix, public PETScObject
  {
  public:

    /// Create empty matrix
    PETScMatrix();

    /// Create M x N matrix
    PETScMatrix(uint M, uint N);

    /// Copy constructor
    explicit PETScMatrix(const PETScMatrix& A);

    /// Create matrix from given PETSc Mat pointer
    explicit PETScMatrix(boost::shared_ptr<Mat> A);

    /// Destructor
    virtual ~PETScMatrix();

    //--- Implementation of the GenericTensor interface ---

    /// Initialize zero tensor using sparsity pattern
    virtual void init(const GenericSparsityPattern& sparsity_pattern);

    /// Return copy of tensor
    virtual PETScMatrix* copy() const;

    /// Return size of given dimension
    virtual uint size(uint dim) const;

    /// Set all entries to zero and keep any sparse structure
    virtual void zero();

    /// Finalize assembly of tensor
    virtual void apply(std::string mode);

    /// Return informal string representation (pretty-print)
    virtual std::string str(bool verbose) const;

    //--- Implementation of the GenericMatrix interface --

    /// Resize matrix to M x N
    virtual void resize(uint M, uint N);

    /// Get block of values
    virtual void get(double* block, uint m, const uint* rows, uint n, const uint* cols) const;

    /// Set block of values
    virtual void set(const double* block, uint m, const uint* rows, uint n, const uint* cols);

    /// Add block of values
    virtual void add(const double* block, uint m, const uint* rows, uint n, const uint* cols);

    /// Add multiple of given matrix (AXPY operation)
    virtual void axpy(double a, const GenericMatrix& A, bool same_nonzero_pattern);

    /// Get non-zero values of given row
    virtual void getrow(uint row, std::vector<uint>& columns, std::vector<double>& values) const;

    /// Set values for given row
    virtual void setrow(uint row, const std::vector<uint>& columns, const std::vector<double>& values);

    /// Set given rows to zero
    virtual void zero(uint m, const uint* rows);

    /// Set given rows to identity matrix
    virtual void ident(uint m, const uint* rows);

    // Matrix-vector product, y = Ax
    virtual void mult(const GenericVector& x, GenericVector& y) const;

    // Matrix-vector product, y = A^T x
    virtual void transpmult(const GenericVector& x, GenericVector& y) const;

    /// Multiply matrix by given number
    virtual const PETScMatrix& operator*= (double a);

    /// Divide matrix by given number
    virtual const PETScMatrix& operator/= (double a);

    /// Assignment operator
    virtual const GenericMatrix& operator= (const GenericMatrix& A);

    //--- Special functions ---

    /// Return linear algebra backend factory
    virtual LinearAlgebraFactory& factory() const;

    //--- Special PETScFunctions ---

    /// Return PETSc Mat pointer
    boost::shared_ptr<Mat> mat() const;

    /// Return norm of matrix
    double norm(std::string norm_type) const;

    /// Assignment operator
    const PETScMatrix& operator= (const PETScMatrix& A);

  private:

    // PETSc norm types
    static const std::map<std::string, NormType> norm_types;

    // PETSc Mat pointer
    boost::shared_ptr<Mat> A;

  };

}

#endif

#endif
