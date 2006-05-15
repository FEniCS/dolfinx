// Copyright (C) 2004-2006 Johan Hoffman, Johan Jansson and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Garth N. Wells 2005, 2006.
//
// First added:  2004
// Last changed: 2006-05-15

#ifndef __SPARSE_VECTOR_H
#define __SPARSE_VECTOR_H

#ifdef HAVE_PETSC_H

#include <dolfin/constants.h>
#include <dolfin/dolfin_log.h>
#include <dolfin/Variable.h>
#include <dolfin/PETScManager.h>
#include <dolfin/GenericVector.h>

namespace dolfin
{
  
  class SparseVectorElement;

  /// This class represents a sparse vector of dimension N.
  /// It is a simple wrapper for a PETSc vector pointer (Vec).
  ///
  /// The interface is intentionally simple. For advanced usage,
  /// access the PETSc Mat pointer using the function mat() and
  /// use the standard PETSc interface.

  class SparseVector : public GenericVector, public Variable
  {
  public:

    /// Empty vector
    SparseVector();

    /// Create vector of given size
    SparseVector(uint size);

    /// Create vector from given PETSc Vec pointer
    SparseVector(Vec x);

    /// Copy constructor
    SparseVector(const SparseVector& x);
    
    /// Destructor
    ~SparseVector ();

    /// Initialize vector data
    void init(uint size);

    /// Clear vector data
    void clear();

    /// Return size of vector
    uint size() const;

    /// Return PETSc Vec pointer
    Vec vec() const;

    /// Return array containing this processor's portion of the data.
    /// After usage, the function restore() must be called.
    real* array();

    /// Return array containing this processor's portion of the data.
    /// After usage, the function restore() must be called. (const version)
    const real* array() const;

    /// Restore array after a call to array()
    void restore(real data[]);

    /// Restore array after a call to array(), const version
    void restore(const real data[]) const;

    /// Addition (AXPY)
    void axpy(const real a, const SparseVector& x) const;

    /// Element-wise division
    void div(const SparseVector& x);

    /// Element-wise multiplication
    void mult(const SparseVector& x);

    /// Set block of values
    void set(const real block[], const int pos[], int n);

    /// Add block of values
    void add(const real block[], const int pos[], int n);

    /// Get block of values from vector
    void get(real block[], const int cols[], int n) const;

    /// Apply changes to vector
    void apply();

    /// Set all entries to zero
    void zero();

    /// Element assignment/access operator
    SparseVectorElement operator() (uint i);

    /// Element access operator for a const SparseVector
    real operator() (uint i) const;

    /// Assignment of vector
    const SparseVector& operator= (const SparseVector& x);

    /// Assignment of all elements to a single scalar value
    const SparseVector& operator= (real a);

    /// Add vector x
    const SparseVector& operator+= (const SparseVector& x);

    /// Subtract vector x
    const SparseVector& operator-= (const SparseVector& x);

    /// Multiply vector with scalar
    const SparseVector& operator*= (real a);

    /// Divide vector by scalar
    const SparseVector& operator/= (real a);

    /// Scalar product
    real operator*(const SparseVector& x);

    /// Compute norm of vector
    enum NormType { l1, l2, linf };
    real norm(NormType type = l2) const;

    /// Compute sum of vector
    real sum() const;

    /// Return value of maximum component of vector
    real max() const;
    
    /// Return value of minimum component of vector
    real min() const;
    
    /// Display vector
    void disp() const;

    /// Output
    friend LogStream& operator<< (LogStream& stream, const SparseVector& A);

    // Friends
    friend class SparseMatrix;
    friend class SparseVectorElement;

    // Element access
    real getval(uint i) const;

    // Set value of element
    void setval(uint i, const real a);

    // Add value to element
    void addval(uint i, const real a);

  private:

    // PETSc Vec pointer
    Vec x;
    
    // True if the pointer is a copy of someone else's data
    bool copy;

  };

  /// Reference to an element of the vector
  
  class SparseVectorElement
  {
  public:
    SparseVectorElement(uint i, SparseVector& x);
    SparseVectorElement(const SparseVectorElement& e);
    operator real() const;
    const SparseVectorElement& operator=(const SparseVectorElement& e);
    const SparseVectorElement& operator=(const real a);
    const SparseVectorElement& operator+=(const real a);
    const SparseVectorElement& operator-=(const real a);
    const SparseVectorElement& operator*=(const real a);
  protected:
    uint i;
    SparseVector& x;
  };

}

#endif

#endif
