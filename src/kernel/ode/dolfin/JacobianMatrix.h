// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __JACOBIAN_MATRIX_H
#define __JACOBIAN_MATRIX_H

#include <dolfin/Matrix.h>

namespace dolfin
{
  
  class RHS;

  /// This class represents the Jacobian matrix of the system of
  /// equations defined on a time slab.

  class JacobianMatrix : public Matrix
  {
  public:

    /// Constructor
    JacobianMatrix(RHS& f);

    /// Destructor
    ~JacobianMatrix();

    /// Return dimension of matrix
    unsigned int size(unsigned int dim) const;
    
    /// Multiplication with vector (use with GMRES solver)
    void mult(const Vector& x, Vector& Ax) const;

    /// Recompute at given time for given number of unknowns
    void update(real t, unsigned n);

  private:

    // The right-hand side
    RHS& f;

    // A (sparse) Matrix storing the Jacobian of f
    Matrix dfdu;

    // Number of unknowns
    unsigned int n;
    
  };

}

#endif
