// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __JACOBIAN_MATRIX_H
#define __JACOBIAN_MATRIX_H

#include <dolfin/Matrix.h>

namespace dolfin
{
  
  class ODE;

  /// This class represents the Jacobian matrix of the system of
  /// equations defined on a time slab.

  class JacobianMatrix : public Matrix
  {
  public:

    /// Constructor
    JacobianMatrix(ODE& ode);

    /// Destructor
    ~JacobianMatrix();

    /// Recompute the Jacobian of the right-hand side
    void update(real t);
    
    /// Multiplication with vector (use with GMRES solver)
    void mult(const Vector& x, Vector& Ax) const;

  private:

    Matrix dfdu;
    
  };

}

#endif
