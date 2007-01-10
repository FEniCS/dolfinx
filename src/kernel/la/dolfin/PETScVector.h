// Copyright (C) 2004-2006 Johan Hoffman, Johan Jansson and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Garth N. Wells 2005, 2006.
//
// First added:  2004
// Last changed: 2006-12-12

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
  
  class uBlasVector;
  class PETScVectorElement;

  /// This class represents a vector of dimension N.
  /// It is a simple wrapper for a PETSc vector pointer (Vec).
  ///
  /// The interface is intentionally simple. For advanced usage,
  /// access the PETSc Vec pointer using the function vec() and
  /// use the standard PETSc interface.

  class PETScVector : public GenericVector, public Variable
  {
  public:

    /// Empty vector
    PETScVector();

    /// Create vector of given size
    PETScVector(const uint N);

    /// Create vector from given PETSc Vec pointer
    PETScVector(Vec x);

    /// Copy constructor
    PETScVector(const PETScVector& x);
    
    /// Destructor
    ~PETScVector ();

    /// Initialize vector data
    void init(const uint N);

    /// Clear vector data
    void clear();

    /// Return size of vector
    uint size() const;

    /// Access element value
    real get(const uint i) const;

    /// Set element value
    void set(const uint i, const real value);
    
    // Add value to element
    void add(const uint i, const real value);

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
    void axpy(const real a, const PETScVector& x);

    /// Element-wise division
    void div(const PETScVector& x);

    /// Element-wise multiplication
    void mult(const PETScVector& x);

    /// Element-wise multiplication
    void mult(const real a);

    /// Set block of values
    void set(const real block[], const int pos[], const int n);

    /// Add block of values
    void add(const real block[], const int pos[], const int n);

    /// Get block of values from vector
    void get(real block[], const int cols[], const int n) const;

    /// Apply changes to vector
    void apply();

    /// Set all entries to zero
    void zero();

    /// Element assignment/access operator
    PETScVectorElement operator() (const uint i);

    /// Element access operator for a const PETScVector
    real operator() (const uint i) const;

    /// Assignment of vector
    const PETScVector& operator= (const PETScVector& x);

    /// Assignment of all elements to a single scalar value
    const PETScVector& operator= (const real a);

    /// Add vector x
    const PETScVector& operator+= (const PETScVector& x);

    /// Subtract vector x
    const PETScVector& operator-= (const PETScVector& x);

    /// Multiply vector with scalar
    const PETScVector& operator*= (const real a);

    /// Divide vector by scalar
    const PETScVector& operator/= (const real a);

    /// Scalar product
    real operator*(const PETScVector& x);

    /// Compute norm of vector
    enum NormType { l1, l2, linf };
    real norm(const NormType type = l2) const;

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
    friend class PETScMatrix;
    friend class PETScVectorElement;

    // Create Scatterer
    static VecScatter* createScatterer(PETScVector& x1, PETScVector& x2,
				       const int offset, const int size);

    // Gather x1 (subvector) into x2
    static void gather(PETScVector& x1, PETScVector& x2, VecScatter& x1sc);

    // Scatter part of x2 into x1 (subvector)
    static void scatter(PETScVector& x1, PETScVector& x2, VecScatter& x1sc);

    // Copy values from array into vector
    static void fromArray(const real u[], PETScVector& x, const uint offset,
			  const uint size);

    // Copy values from vector into array
    static void toArray(real y[], const PETScVector&x, const uint offset, 
                  const uint size);

    // Copy values between different vector representations
    void copy(const PETScVector& y, const uint off1, const uint off2, const uint len);
    void copy(const uBlasVector& y, const uint off1, const uint off2, const uint len);

  private:

    // PETSc Vec pointer
    Vec x;
    
    // True if the pointer is a copy of someone else's data
    bool _copy;

  };

  /// Reference to an element of the vector
  
  class PETScVectorElement
  {
  public:
    PETScVectorElement(const uint i, PETScVector& x);
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
