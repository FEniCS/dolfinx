// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __NEW_PDE_H
#define __NEW_PDE_H

#include <dolfin/constants.h>
#include <dolfin/NewArray.h>

namespace dolfin
{

  /// EXPERIMENTAL: Redesign of the evaluation of variational forms

  class NewPDE
  {
  public:

    /// Constructor
    NewPDE();

    /// Destructor
    virtual ~NewPDE();

    /// Evaluate left-hand side (compute element stiffness matrix)
    virtual void lhs(NewArray< NewArray<real> >& A);

    /// Evaluate right-hand side (compute element load vector)
    virtual void rhs(NewArray<real>& b);

  protected:

    real det;
    real g00, g01, g02, g10, g11, g12, g20, g21, g22;

  };

}

#endif
