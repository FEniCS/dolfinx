// Copyright (C) 2005-2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2005
// Last changed: 2006-07-07

#ifndef __HOMOTOPY_JACOBIAN_H
#define __HOMOTOPY_JACOBIAN_H

#include <dolfin/constants.h>
#include <dolfin/uBlasKrylovMatrix.h>

namespace dolfin
{

  class ComplexODE;

  /// This class implements a matrix-free Jacobian for a homotopy
  /// system. It uses the fact that the Jacobian is already
  /// implemented for the ODE system so we don't have to worry about
  /// the translation between real and complex vectors.

  class HomotopyJacobian : public uBlasKrylovMatrix
  {
  public:

    /// Constructor
    HomotopyJacobian(ComplexODE& ode, uBlasVector& u);

    /// Destructor
    ~HomotopyJacobian();

    /// Return number of rows (dim = 0) or columns (dim = 1)
    uint size(uint dim) const;

    /// Compute product y = Ax
    void mult(const uBlasVector& x, uBlasVector& y) const;

  private:
    
    // The ODE for the homotopy system
    ComplexODE& ode;

    // Current solution to linearize around
    uBlasVector& u;

  };

}

#endif
