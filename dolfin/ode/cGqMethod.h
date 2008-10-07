// Copyright (C) 2003-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2005-05-02
// Last changed: 2008-10-07

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
    double ueval(double x0, double values[], double tau) const;

    /// Evaluate solution at given node (inline optimized)
    inline double ueval(double x0, double values[], uint i) const
    { return ( i == 0 ? x0 : values[i - 1] ); }

    /// Compute residual at right end-point    
    double residual(double x0, double values[], double f, double k) const;

    /// Compute new time step based on the given residual
    double timestep(double r, double tol, double k0, double kmax) const;

    /// Compute error estimate (modulo stability factor)
    double error(double k, double r) const;

    /// Display method data
    void disp() const;
    
  protected:

    void computeQuadrature();
    void computeBasis();
    void computeWeights();

  };

}

#endif
