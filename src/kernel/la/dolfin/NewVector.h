// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __NEW_VECTOR_H
#define __NEW_VECTOR_H

#include <petsc/petscvec.h>
#include <dolfin/constants.h>
#include <dolfin/dolfin_log.h>

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
    NewVector(int size);
    NewVector(const NewVector& x);
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

    /// Element access operator
    real operator()(unsigned int i) const;

    /// Element assignment operator
    Index operator()(unsigned int i);

    class Index
    {
    public:
      Index(unsigned int i, NewVector &v);

      void operator=(const real r);

    protected:
      unsigned int i;
      NewVector &v;
    };


  protected:
    // PETSc Vec pointer
    Vec v;
    unsigned int n;

  };


}

#endif
