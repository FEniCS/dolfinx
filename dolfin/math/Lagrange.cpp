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
// Modified by Garth N. Wells, 2006.
// Modified by Benjamin Kehlet, 2009
//
// First added:  2003-06-12
// Last changed: 2009-03-20

#include <cmath>
#include <dolfin/common/constants.h>
#include "Lagrange.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
Lagrange::Lagrange(std::size_t q) : _q(q), counter(0), points(q + 1, 0.0),
   instability_detected("Warning: Lagrange polynomial is not numerically stable. The degree is too high.")
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Lagrange::Lagrange(const Lagrange& p) : _q(p._q), counter(p.counter),
    points(p.points), instability_detected(p.instability_detected)
{
  if (counter == size())
    init();
}
//-----------------------------------------------------------------------------
void Lagrange::set(std::size_t i, double x)
{
  dolfin_assert(i <= _q);

  points[i] = x;

  counter++;
  if (counter == size())
    init();
}
//-----------------------------------------------------------------------------
std::size_t Lagrange::size() const
{
  return points.size();
}
//-----------------------------------------------------------------------------
std::size_t Lagrange::degree() const
{
  return _q;
}
//-----------------------------------------------------------------------------
double Lagrange::point(std::size_t i) const
{
  dolfin_assert(i < points.size());
  return points[i];
}
//-----------------------------------------------------------------------------
double Lagrange::operator() (std::size_t i, double x)
{
  return eval(i, x);
}
//-----------------------------------------------------------------------------
double Lagrange::eval(std::size_t i, double x)
{
  dolfin_assert(i < points.size());

  double product(constants[i]);
  for (std::size_t j = 0; j < points.size(); j++)
  {
    if (j != i)
      product *= x - points[j];
  }

  return product;
}
//-----------------------------------------------------------------------------
double Lagrange::ddx(std::size_t i, double x)
{
  dolfin_assert(i < points.size());

  double s(0.0);
  double prod(1.0);
  bool x_equals_point = false;

  for (std::size_t j = 0; j < points.size(); ++j)
  {
    if (j != i)
    {
      double t = x - points[j];
      if (std::abs(t) < DOLFIN_EPS)
        x_equals_point = true;
      else
      {
        s += 1/t;
        prod *= t;
      }
    }
  }

  if (x_equals_point)
    return prod*constants[i];
  else
    return prod*constants[i]*s;
}
//-----------------------------------------------------------------------------
double Lagrange::dqdx(std::size_t i)
{
  double product = constants[i];
  for (std::size_t j = 1; j <= _q; j++)
    product *= (double) j;

  return product;
}
//-----------------------------------------------------------------------------
std::string Lagrange::str(bool verbose) const
{
  std::stringstream s;

  if (verbose)
  {
    s << str(false) << std::endl << std::endl;
    for (std::size_t i = 0; i < points.size(); i++)
      s << "  x[" << i << "] = " << points[i];
  }
  else
  {
    s << "<Lagrange polynomial of degree " << _q << " with "
          << points.size() << " points>";
  }

  return s.str();
}
//-----------------------------------------------------------------------------
void Lagrange::init()
{
  // Note: this will be computed when n nodal points have been set,
  // assuming they are distinct. Precomputing the constants has a downside
  // wrt to numerical stability, since the constants will decrease as
  // the degree increases (and for high order be less than epsilon.

  constants.resize(points.size());

  // Compute constants
  for (std::size_t i = 0; i < points.size(); i++)
  {
    double product = 1.0;
    for (std::size_t j = 0; j < points.size(); j++)
    {
      if (j != i)
      {
        if (std::abs(points[i] - points[j]) < DOLFIN_EPS)
        {
          dolfin_error("Lagrange.cpp",
                       "create Lagrange polynomial",
                       "Nodal points are not distinct");
        }
        product *= points[i] - points[j];
      }
    }

    if (std::abs(product) < DOLFIN_EPS)
      instability_detected();

    constants[i] = 1.0/product;
  }
}
//-----------------------------------------------------------------------------
