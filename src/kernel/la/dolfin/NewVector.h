// Copyright (C) 2004 Johan Jansson.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg, 2005.

#ifndef __NEW_VECTOR_H
#define __NEW_VECTOR_H

#include <petsc/petscvec.h>
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

  class NewVector
  {
  public:

    class Index;

    /// Empty vector
    NewVector();

    /// Create vector of given size
    NewVector(uint size);

    /// Copy constructor
    NewVector(const NewVector& x);

    /// Create vector from old vector (will be removed)
    NewVector(const Vector& x);

    /// Destructor
    ~NewVector ();

    /// Initialize vector data
    void init(uint size);

    /// Clear vector data
    void clear();

    /// Return size of vector
    uint size() const;

    /// Return PETSc Vec pointer
    Vec vec();

    /// Return PETSc Vec pointer
    const Vec vec() const;

    /// Return a contiguous array containing this processor's portion
    /// of the data. After usage, the function restore() must be
    /// called.
    real* array();

    /// Restore array after a call to array()
    void restore(real data[]);

    /// Addition (AXPY)
    void add(const real a, const NewVector& x) const;

    /// Element assignment operator
    Index operator() (uint i);

    /// Assignment of all elements to a single scalar value
    const NewVector& operator= (real a);

    /// Display vector
    void disp() const;

    /// Reference to a position in the vector
    class Index
    {
    public:
      Index(uint i, NewVector& v);
      operator real() const;
      void operator=(const real r);
    protected:
      uint i;
      NewVector &v;
    };

  protected:

    // Element assignment
    void setvalue(uint i, const real r);

    // Element access
    real getvalue(uint i) const;

  private:

    // PETSc Vec pointer
    Vec v;
    
  };
}

#endif
