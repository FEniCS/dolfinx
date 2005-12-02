// Copyright (C) 2004-2005 Johan Hoffman, Johan Jansson and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Garth N. Wells 2005.
//
// First added:  2004
// Last changed: 2005-11-01

#ifndef __VECTOR_H
#define __VECTOR_H

#include <petscvec.h>
#include <dolfin/constants.h>
#include <dolfin/dolfin_log.h>
#include <dolfin/Vector.h>

namespace dolfin
{
  
  /// This class represents a vector of dimension n. It is a
  /// simple wrapper for a PETSc vector (Vec). The interface is
  /// intentionally simple. For advanced usage, access the PETSc Vec
  /// pointer using the function vec() and use the standard PETSc
  /// interface.

  class VectorElement;

  class Vector
  {
  public:


    /// Empty vector
    Vector();

    /// Create vector of given size
    Vector(uint size);

    /// Create vector from given PETSc Vec pointer
    Vector(Vec x);

    /// Copy constructor
    Vector(const Vector& x);
    
    /// Destructor
    ~Vector ();

    /// Initialize vector data
    void init(uint size);

    /// Clear vector data
    void clear();

    /// Return size of vector
    uint size() const;

    /// Return PETSc Vec pointer
    Vec vec();

    /// Return PETSc Vec pointer, const version
    const Vec vec() const;

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
    void axpy(const real a, const Vector& x) const;

    /// Add block of values to vector
    void add(const real block[], const int cols[], int n); 

    /// Apply changes to vector
    void apply();

    /// Element assignment/access operator
    VectorElement operator() (uint i);

    /// Element access operator for a const Vector
    real operator() (uint i) const;

    /// Assignment of vector
    const Vector& operator= (const Vector& x);

    /// Assignment of all elements to a single scalar value
    const Vector& operator= (real a);

    /// Add vector x
    const Vector& operator+= (const Vector& x);

    /// Subtract vector x
    const Vector& operator-= (const Vector& x);

    /// Multiply vector with scalar
    const Vector& operator*= (real a);

    /// Divide vector by scalar
    const Vector& operator/= (real a);

    /// Scalar product
    real operator*(const Vector& x);

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
    friend LogStream& operator<< (LogStream& stream, const Vector& A);

    // Friends
    friend class Matrix;
    friend class VectorElement;

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
  
  class VectorElement
  {
  public:
    VectorElement(uint i, Vector& x);
    VectorElement(const VectorElement& e);
    operator real() const;
    const VectorElement& operator=(const VectorElement& e);
    const VectorElement& operator=(const real a);
    const VectorElement& operator+=(const real a);
    const VectorElement& operator-=(const real a);
    const VectorElement& operator*=(const real a);
  protected:
    uint i;
    Vector& x;
  };

}

#endif
