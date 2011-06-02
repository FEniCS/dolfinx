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
// First added:  2003-06-03
// Last changed: 2009-08-11

#include <iomanip>
#include <cmath>
#include <dolfin/common/constants.h>
#include <dolfin/common/real.h>
#include <dolfin/log/dolfin_log.h>
#include <dolfin/math/Legendre.h>
#include "GaussQuadrature.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
GaussQuadrature::GaussQuadrature(unsigned int n) : GaussianQuadrature(n)
{
  init();

  if (!check(2*n-1))
    error("Gauss quadrature not ok, check failed.");
}
//-----------------------------------------------------------------------------
std::string GaussQuadrature::str(bool verbose) const
{
  std::stringstream s;

  if (verbose)
  {
    s << str(false) << std::endl << std::endl;

    s << " i    points                   weights" << std::endl;
    s << "-----------------------------------------------------" << std::endl;

    s << std::setiosflags(std::ios::scientific) << std::setprecision(16);

    for (uint i = 0; i < points.size(); i++)
    {
      s << i << " "
        << to_double(points[i]) << " "
        << to_double(weights[i]) << " "
        << std::endl;
    }
  }
  else
  {
    s << "<GaussQuadrature with " << points.size() << " points on [-1, 1]>";
  }

  return s.str();
}
//----------------------------------------------------------------------------
void GaussQuadrature::compute_points()
{
  // Compute Gauss quadrature points on [-1,1] as the
  // as the zeroes of the Legendre polynomials using Newton's method

  const uint n = points.size();

  // Special case n = 1
  if (n == 1)
  {
    points[0] = 0.0;
    return;
  }

  Legendre p(n);
  real x, dx;

  // Compute the points by Newton's method
  for (unsigned int i = 0; i <= ((n-1)/2); i++)
  {

    // Initial guess
    x = cos(DOLFIN_PI*(double(i+1)-0.25)/(double(n)+0.5));

    // Newton's method
    do
    {
      dx = - p(x) / p.ddx(x);
      x  = x + dx;
    } while (real_abs(dx) > real_epsilon());

    // Save the value using the symmetry of the points
    points[i] = - x;
    points[n-1-i] = x;
  }

  // Set middle node
  if ((n % 2) != 0)
    points[n/2] = 0.0;
}
//-----------------------------------------------------------------------------
