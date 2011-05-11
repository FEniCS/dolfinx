// Copyright (C) 2003-2009 Anders Logg
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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2003-06-03
// Last changed: 2009-09-08

#ifndef __GAUSS_QUADRATURE_H
#define __GAUSS_QUADRATURE_H

#include <dolfin/log/dolfin_log.h>
#include "GaussianQuadrature.h"

namespace dolfin
{

  /// Gauss (Gauss-Legendre) quadrature on the interval [-1,1].
  /// The n quadrature points are given by the zeros of the
  /// n:th Legendre Pn(x).
  ///
  /// The quadrature points are computed using Newton's method, and
  /// the quadrature weights are computed by solving a linear system
  /// determined by the condition that Gauss quadrature with n points
  /// should be exact for polynomials of degree 2n-1.

  class GaussQuadrature : public GaussianQuadrature
  {
  public:

    /// Create Gauss quadrature with n points
    GaussQuadrature(unsigned int n);

    /// Return informal string representation (pretty-print)
    std::string str(bool verbose) const;

  private:

    void compute_points();

  };

}

#endif
