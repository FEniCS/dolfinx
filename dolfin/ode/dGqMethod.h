// Copyright (C) 2003-2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2005-05-02
// Last changed: 2006-07-07

#ifndef __DGQ_METHOD_H
#define __DGQ_METHOD_H

#include "Method.h"

namespace dolfin
{

  /// Contains all numeric constants, such as nodal points and nodal weights,
  /// needed for the dG(q) method. The order q must be at least 0. Note that
  /// q refers to the polynomial order and not the order of convergence for
  /// the method, which is 2q + 1.

  class dGqMethod : public Method
  {
  public:
    
    dGqMethod(unsigned int q);

    /// Evaluate solution at given point
    double ueval(double x0, double values[], double tau) const;

    /// Evaluate solution at given node (inline optimized)
    double ueval(double x0, double values[], uint i) const
    { return values[i]; }

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
