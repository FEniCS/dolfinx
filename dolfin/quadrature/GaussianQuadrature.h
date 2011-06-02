// Copyright (C) 2003-2005 Anders Logg
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
// First added:  2003-06-03
// Last changed: 2009-08-10

#ifndef __GAUSSIAN_QUADRATURE_H
#define __GAUSSIAN_QUADRATURE_H

#include "Quadrature.h"

namespace dolfin {

  /// Gaussian-type quadrature rule on the double line,
  /// including Gauss, Radau, and Lobatto quadrature.
  ///
  /// Points and weights are computed to be exact within a tolerance
  /// of DOLFIN_EPS. Comparing with known exact values for n <= 3 shows
  /// that we obtain full precision (16 digits, error less than 2e-16).

  class GaussianQuadrature : public Quadrature
  {
  public:

    GaussianQuadrature(unsigned int n);

  protected:

    // Compute points and weights
    void init();

    // Compute quadrature points
    virtual void compute_points() = 0;

    // Compute quadrature weights
    void compute_weights();

    // Check that quadrature is exact for given degree q
    bool check(unsigned int q) const;

  };

}

#endif
