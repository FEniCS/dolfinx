// Copyright (C) 2004 Johan Jansson.
// Licensed under the GNU GPL Version 2.

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

    NewVector();
    NewVector(unsigned int size);
    NewVector(const NewVector& x);
    NewVector(const Vector& x);
    ~NewVector ();

    /// Initialize vector data
    void init(unsigned int size);

    /// Clear vector data
    void clear();

    /// Return size of vector
    unsigned int size() const;

    /// Display vector
    void disp() const;

    /// Return PETSc Vec pointer
    Vec vec();

    /// Return PETSc Vec pointer
    const Vec vec() const;

    /// Element assignment
    void setvalue(int i, const real r);

    /// Element access
    real getvalue(int i) const;

    /// Element assignment operator
    Index operator()(int i);

    class Index
    {
    public:
      Index(int i, NewVector &v);

      operator real() const;
      void operator=(const real r);

    protected:
      int i;
      NewVector &v;
    };


  protected:
    // PETSc Vec pointer
    Vec v;
  };
}

#endif
