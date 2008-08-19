// Copyright (C) 2003-2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2005-05-02
// Last changed: 2006-07-07

#ifndef __CGQ_METHOD_H
#define __CGQ_METHOD_H

#include "Method.h"

namespace dolfin
{

  /// Contains all numeric constants, such as nodal points and nodal weights,
  /// needed for the cG(q) method. The order q must be at least 1. Note that
  /// q refers to the polynomial order and not the order of convergence for
  /// the method, which is 2q.

  class cGqMethod : public Method
  {
  public:
    
    cGqMethod(unsigned int q);

    /// Evaluate solution at given point
    real ueval(real x0, real values[], real tau) const;

    /// Evaluate solution at given point
    real ueval(real x0, uBLASVector& values, uint offset, real tau) const;

    /// Evaluate solution at given node (inline optimized)
    inline real ueval(real x0, real values[], uint i) const
    { return ( i == 0 ? x0 : values[i - 1] ); }

    /// Compute residual at right end-point    
    real residual(real x0, real values[], real f, real k) const;

    /// Compute residual at right end-point
    real residual(real x0, uBLASVector& values, uint offset, real f, real k) const;

    /// Compute new time step based on the given residual
    real timestep(real r, real tol, real k0, real kmax) const;

    /// Compute error estimate (modulo stability factor)
    real error(real k, real r) const;

    /// Display method data
    void disp() const;
    
  protected:

    void computeQuadrature();
    void computeBasis();
    void computeWeights();

  };

}

#endif
