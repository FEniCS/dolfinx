// Copyright (C) 2004-2006 Johan Hoffman, Johan Jansson and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Garth N. Wells 2005, 2006.
//
// First added:  2004
// Last changed: 2006-05-31

#ifndef __PETSC_VECTOR_H
#define __PETSC_VECTOR_H

#ifdef HAVE_PETSC_H

#include <dolfin/constants.h>
#include <dolfin/dolfin_log.h>
#include <dolfin/Variable.h>
#include <dolfin/GenericVector.h>
#include <dolfin/PETScManager.h>

namespace dolfin
{
  
  class PETScVectorElement;

  /// This class represents a vector of dimension N.
  /// It is a simple wrapper for a PETSc vector pointer (Vec).
  ///
  /// The interface is intentionally simple. For advanced usage,
  /// access the PETSc Mat pointer using the function mat() and
  /// use the standard PETSc interface.

  class PETScVector : public GenericVector, public Variable
  {
  public:

    /// Empty vector
    PETScVector();

    /// Create vector of given size
    PETScVector(uint size);

    /// Create vector from given PETSc Vec pointer
    PETScVector(Vec x);

    /// Copy constructor
    PETScVector(const PETScVector& x);
    
    /// Destructor
    ~PETScVector ();

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
    void axpy(const real a, const PETScVector& x) const;

    /// Element-wise division
    void div(const PETScVector& x);

    /// Element-wise multiplication
    void mult(const PETScVector& x);

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
    PETScVectorElement operator() (uint i);

    /// Element access operator for a const PETScVector
    real operator() (uint i) const;

    /// Assignment of vector
    const PETScVector& operator= (const PETScVector& x);

    /// Assignment of all elements to a single scalar value
    const PETScVector& operator= (real a);

    /// Add vector x
    const PETScVector& operator+= (const PETScVector& x);

    /// Subtract vector x
    const PETScVector& operator-= (const PETScVector& x);

    /// Multiply vector with scalar
    const PETScVector& operator*= (real a);

    /// Divide vector by scalar
    const PETScVector& operator/= (real a);

    /// Scalar product
    real operator*(const PETScVector& x);

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
    friend LogStream& operator<< (LogStream& stream, const PETScVector& A);

    // Friends
    friend class PETScSparseMatrix;
    friend class PETScVectorElement;

    // Element access
    real getval(uint i) const;

    // Set value of element
    void setval(uint i, const real a);

    // Add value to element
    void addval(uint i, const real a);

    // Create Scatterer
    static VecScatter* createScatterer(PETScVector& x1, PETScVector& x2,
				       int offset, int size);

    // Gather x1 (subvector) into x2
    static void gather(PETScVector& x1, PETScVector& x2, VecScatter& x1sc);

    // Scatter part of x2 into x1 (subvector)
    static void scatter(PETScVector& x1, PETScVector& x2, VecScatter& x1sc);

    // Copy values from array into vector
    static void fromArray(const real u[], PETScVector& x, uint offset,
			  uint size);

    // Copy values from vector into array
    static void toArray(real y[], PETScVector&x, uint offset, uint size);

  private:

    // PETSc Vec pointer
    Vec x;
    
    // True if the pointer is a copy of someone else's data
    bool copy;

  };

  /// Reference to an element of the vector
  
  class PETScVectorElement
  {
  public:
    PETScVectorElement(uint i, PETScVector& x);
    PETScVectorElement(const PETScVectorElement& e);
    operator real() const;
    const PETScVectorElement& operator=(const PETScVectorElement& e);
    const PETScVectorElement& operator=(const real a);
    const PETScVectorElement& operator+=(const real a);
    const PETScVectorElement& operator-=(const real a);
    const PETScVectorElement& operator*=(const real a);
  protected:
    uint i;
    PETScVector& x;
  };

}

#endif

#endif
