// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __NEW_JACOBIAN_MATRIX_H
#define __NEW_JACOBIAN_MATRIX_H

#include <dolfin/VirtualMatrix.h>

namespace dolfin
{
  
  class ODE;
  class NewTimeSlab;
  class NewMethod;
    
  /// This class represents the Jacobian matrix of the system of
  /// equations defined on a time slab.

  class NewJacobianMatrix : public VirtualMatrix
  {
  public:

    /// Constructor
    NewJacobianMatrix(ODE& ode, NewTimeSlab& timeslab, const NewMethod& method);

    /// Destructor
    ~NewJacobianMatrix();

    /// Compute product y = Ax
    void mult(Vec x, Vec y) const;

    /// Recompute Jacobian at given time
    void update(real t);

  private:

    // The ODE
    ODE& ode;

    // The time slab
    NewTimeSlab& ts;
    
    // Method, mcG(q) or mdG(q)
    const NewMethod& method;

    // Values of the Jacobian df/du of the right-hand side
    real* Jvalues;

    // Indices for first element of each row for the Jacobian df/du
    uint* Jindices;
    
  };

}

#endif
