// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __HOMOTOPY_JACOBIAN_H
#define __HOMOTOPY_JACOBIAN_H

#include <dolfin/constants.h>
#include <dolfin/VirtualMatrix.h>

namespace dolfin
{

  class ComplexODE;
  class NewVector;

  /// This class implements a matrix-free Jacobian for a homotopy
  /// system. It uses the fact that the Jacobian is already
  /// implemented for the ODE system so we don't have to worry about
  /// the translation between real and complex vectors.

  class HomotopyJacobian : public VirtualMatrix
  {
  public:

    /// Constructor
    HomotopyJacobian(ComplexODE& ode, NewVector& u);

    /// Destructor
    ~HomotopyJacobian();

    /// Compute product y = Ax
    void mult(const NewVector& x, NewVector& y) const;

  private:
    
    // The ODE for the homotopy system
    ComplexODE& ode;

    // Current solution to linearize around
    NewVector& u;

  };

}

#endif
