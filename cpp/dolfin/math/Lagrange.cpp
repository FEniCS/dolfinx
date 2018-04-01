// Copyright (C) 2003-2008 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Lagrange.h"
#include <cmath>
#include <dolfin/common/constants.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
dolfin::math::Lagrange::Lagrange(std::size_t q)
    : _q(q), counter(0), points(q + 1, 0.0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
dolfin::math::Lagrange::Lagrange(const Lagrange& p)
    : _q(p._q), counter(p.counter), points(p.points)
{
  if (counter == size())
    init();
}
//-----------------------------------------------------------------------------
void dolfin::math::Lagrange::set(std::size_t i, double x)
{
  assert(i <= _q);

  points[i] = x;

  counter++;
  if (counter == size())
    init();
}
//-----------------------------------------------------------------------------
std::size_t dolfin::math::Lagrange::size() const { return points.size(); }
//-----------------------------------------------------------------------------
std::size_t dolfin::math::Lagrange::degree() const { return _q; }
//-----------------------------------------------------------------------------
double dolfin::math::Lagrange::point(std::size_t i) const
{
  assert(i < points.size());
  return points[i];
}
//-----------------------------------------------------------------------------
double dolfin::math::Lagrange::operator()(std::size_t i, double x)
{
  return eval(i, x);
}
//-----------------------------------------------------------------------------
double dolfin::math::Lagrange::eval(std::size_t i, double x)
{
  assert(i < points.size());

  double product(constants[i]);
  for (std::size_t j = 0; j < points.size(); j++)
  {
    if (j != i)
      product *= x - points[j];
  }

  return product;
}
//-----------------------------------------------------------------------------
double dolfin::math::Lagrange::ddx(std::size_t i, double x)
{
  assert(i < points.size());

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
        s += 1 / t;
        prod *= t;
      }
    }
  }

  if (x_equals_point)
    return prod * constants[i];
  else
    return prod * constants[i] * s;
}
//-----------------------------------------------------------------------------
double dolfin::math::Lagrange::dqdx(std::size_t i)
{
  double product = constants[i];
  for (std::size_t j = 1; j <= _q; j++)
    product *= (double)j;

  return product;
}
//-----------------------------------------------------------------------------
std::string dolfin::math::Lagrange::str(bool verbose) const
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
    s << "<Lagrange polynomial of degree " << _q << " with " << points.size()
      << " points>";
  }

  return s.str();
}
//-----------------------------------------------------------------------------
void dolfin::math::Lagrange::init()
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
          log::dolfin_error("Lagrange.cpp", "create Lagrange polynomial",
                       "Nodal points are not distinct");
        }
        product *= points[i] - points[j];
      }
    }

    //if (std::abs(product) < DOLFIN_EPS)
    //  instability_detected();

    constants[i] = 1.0 / product;
  }
}
//-----------------------------------------------------------------------------
