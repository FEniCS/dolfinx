// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __NEW_VECTOR_H
#define __NEW_VECTOR_H

#include <petsc/petscvec.h>
#include <dolfin/constants.h>
#include <dolfin/dolfin_log.h>

namespace dolfin
{
  
  /// This class represents a matrix of dimension m x n. It is a
  /// simple wrapper for a PETSc matrix (Mat). The interface is
  /// intentionally simple. For advanced usage, access the PETSc Mat
  /// pointer using the function mat() and use the standard PETSc
  /// interface.

  class NewVector
  {
  public:

    NewVector();
    NewVector(int size);
    NewVector(const NewVector& x);
    ~NewVector ();

    void init(unsigned int size);
    void clear();
    unsigned int size() const;

    void disp() const;

  protected:

    // PETSc Mat pointer
    Vec v;
    unsigned int n;
    
  };

}

#endif
