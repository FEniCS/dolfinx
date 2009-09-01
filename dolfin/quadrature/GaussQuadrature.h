// Copyright (C) 2003-2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2003-06-03
// Last changed: 2009-08-10

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
    std::string str(bool verbose=false) const;

  private:

    void compute_points();

  };

}

#endif
