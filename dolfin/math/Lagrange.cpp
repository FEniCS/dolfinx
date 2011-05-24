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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Garth N. Wells, 2006.
// Modified by Benjamin Kehlet, 2009
//
// First added:  2003-06-12
// Last changed: 2009-03-20

#include <dolfin/common/constants.h>
#include <dolfin/common/real.h>
#include "Lagrange.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
Lagrange::Lagrange(unsigned int q)
  :q(q), counter(0), points(q + 1, 0.0),
   instability_detected("Warning: Lagrange polynomial is not numerically stable. The degree is too high.")
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Lagrange::Lagrange(const Lagrange& p)
  : q(p.q), counter(p.counter), points(p.points),
    instability_detected(p.instability_detected)
{
  if (counter == size())
    init();
}
//-----------------------------------------------------------------------------
Lagrange::~Lagrange()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void Lagrange::set(unsigned int i, real x)
{
  assert(i <= q);

  points[i] = x;

  counter++;
  if (counter == size())
    init();
}
//-----------------------------------------------------------------------------
unsigned int Lagrange::size() const
{
  return points.size();
}
//-----------------------------------------------------------------------------
unsigned int Lagrange::degree() const
{
  return q;
}
//-----------------------------------------------------------------------------
real Lagrange::point(unsigned int i) const
{
  assert(i < points.size());
  return points[i];
}
//-----------------------------------------------------------------------------
real Lagrange::operator() (unsigned int i, real x)
{
  return eval(i, x);
}
//-----------------------------------------------------------------------------
real Lagrange::eval(unsigned int i, real x)
{
  assert(i < points.size());

  real product(constants[i]);
  for (unsigned int j = 0; j < points.size(); j++)
  {
    if (j != i)
      product *= x - points[j];
  }

  return product;
}
//-----------------------------------------------------------------------------
real Lagrange::ddx(uint i, real x)
{
  assert(i < points.size());

  real s(0.0);
  real prod(1.0);
  bool x_equals_point = false;

  for (uint j = 0; j < points.size(); ++j)
  {
    if (j != i)
    {
      real t = x - points[j];
      if (real_abs(t) < real_epsilon())
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
real Lagrange::dqdx(unsigned int i)
{
  real product = constants[i];
  for (unsigned int j = 1; j <= q; j++)
    product *= (real) j;

  return product;
}
//-----------------------------------------------------------------------------
std::string Lagrange::str(bool verbose) const
{
  std::stringstream s;

  if (verbose)
  {
    s << str(false) << std::endl << std::endl;
    for (unsigned int i = 0; i < points.size(); i++)
      s << "  x[" << i << "] = " << to_double(points[i]);
  }
  else
    s << "<Lagrange polynomial of degree " << q << " with " << points.size() << " points>";

  return s.str();
}
//-----------------------------------------------------------------------------
void Lagrange::init()
{
  // Note: this will be computed when n nodal points have been set, assuming they are
  // distinct. Precomputing the constants has a downside wrt. to numerical stability, since
  // the constants will decrease as the degree increases (and for high order be less than
  // epsilon.

  constants.resize(points.size());

  // Compute constants
  for (unsigned int i = 0; i < points.size(); i++)
  {
    real product = 1.0;
    for (unsigned int j = 0; j < points.size(); j++)
    {
      if (j != i)
      {
        if (real_abs(points[i] - points[j]) < real_epsilon())
          error("Lagrange points not distinct");
        product *= points[i] - points[j];
      }
    }

    if (real_abs(product) < real_epsilon())
      instability_detected();

    constants[i] = 1.0/product;
  }
}
//-----------------------------------------------------------------------------
