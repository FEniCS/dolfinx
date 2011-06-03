// Copyright (C) 2003-2008 Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Benjamin Kehlet 2009
//
// First added:  2005-05-02
// Last changed: 2009-09-08

#ifndef __CGQ_METHOD_H
#define __CGQ_METHOD_H

#include <dolfin/common/real.h>
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

    /// Evaluate solution at given node (inline optimized)
    inline real ueval(real x0, real values[], uint i) const
    { return ( i == 0 ? x0 : values[i - 1] ); }

    /// Compute residual at right end-point
    real residual(real x0, real values[], real f, real k) const;

    /// Compute new time step based on the given residual
    real timestep(real r, real tol, real k0, real kmax) const;

    /// Compute error estimate (modulo stability factor)
    real error(real k, real r) const;

    /// Replace the solution values with the nodal values solution polynomial
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
