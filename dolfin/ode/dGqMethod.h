// Copyright (C) 2003-2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Benjamin Kehlet 2009
//
// First added:  2005-05-02
// Last changed: 2010-03-25

#ifndef __DGQ_METHOD_H
#define __DGQ_METHOD_H

#include <dolfin/common/real.h>
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
    real ueval(real x0, real values[], real tau) const;

    /// Evaluate solution at given node (inline optimized)
    real ueval(real, real values[], uint i) const
    { return values[i]; }

    /// Compute residual at right end-point
    real residual(real x0, real values[], real f, real k) const;

    /// Compute new time step based on the given residual
    real timestep(real r, real tol, real k0, real kmax) const;

    /// Compute error estimate (modulo stability factor)
    real error(real k, real r) const;

    void get_nodal_values(const real& x0, const real* x, real* nodal_values) const;

    /// Return informal string representation (pretty-print)
    std::string str(bool verbose) const;

  protected:

    void compute_quadrature();
    void compute_basis();
    void compute_weights();

  };

}

#endif



