// Copyright (C) 2003-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2006.
// Modified by Benjamin Kehlet, 2009
//
// First added:  2003-06-12
// Last changed: 2009-03-20

#include <dolfin/common/constants.h>
#include <dolfin/common/real.h>
#include <dolfin/log/dolfin_log.h>
#include "Lagrange.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
Lagrange::Lagrange(unsigned int q)
{
  this->q = q;
  n = q + 1;

  points = new real[n];
  for (unsigned int i = 0; i < n; i++)
    points[i] = 0.0;

  constants = 0;
  init();
}
//-----------------------------------------------------------------------------
Lagrange::Lagrange(const Lagrange& p)
{
  this->q = p.q;
  n = p.n;

  points = new real[p.n];
  for (unsigned int i = 0; i < p.n; i++)
    points[i] = p.points[i];

  constants = 0;
  init();
}
//-----------------------------------------------------------------------------
Lagrange::~Lagrange()
{
  if ( points )
    delete [] points;
  points = 0;

  if ( constants )
    delete [] constants;
  constants = 0;
}
//-----------------------------------------------------------------------------
void Lagrange::set(unsigned int i, real x)
{
  assert(i <= q);

  points[i] = x;
  init();
}
//-----------------------------------------------------------------------------
unsigned int Lagrange::size() const
{
  return n;
}
//-----------------------------------------------------------------------------
unsigned int Lagrange::degree() const
{
  return q;
}
//-----------------------------------------------------------------------------
real Lagrange::point(unsigned int i) const
{
  assert(i <= q);

  return points[i];
}
//-----------------------------------------------------------------------------
real Lagrange::operator() (unsigned int i, real x)
{
  return eval(i,x);
}
//-----------------------------------------------------------------------------
real Lagrange::eval(unsigned int i, real x)
{
  assert(i <= q);

  real product(constants[i]);
  for (unsigned int j = 0; j < n; j++)
    if ( j != i )
      product *= x - points[j];

  return product;
}
//-----------------------------------------------------------------------------
real Lagrange::ddx(uint i, real x)
{
  assert(i <= q);
  
  real s(0.0);
  real prod(1.0);
  bool x_equals_point = false;

  for (uint j = 0; j < n; ++j) 
  {
    if (j != i) 
    {
      real t = x - points[j];
      if (real_abs(t) < real_epsilon()) 
      {
	x_equals_point = true;
      } else 
      {
	s += 1/t;
	prod *= t;
      }
    }
  }

  if (x_equals_point) return prod*constants[i];
  else                return prod*constants[i]*s;
  
  /*
  real sum(0);
  for (uint j = 0; j < n; j++) {
    if ( j != i ) {
      real product = 1.0;
      for (uint k = 0; k < n; k++)
	if ( k != i && k != j )
	  product *= x - points[k];
      sum += product;
    }
  }

  return sum * constants[i];
  */
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

    for (unsigned int i = 0; i < n; i++)
      s << "  x[" << i << "] = " << to_double(points[i]);
  }
  else
  {
    s << "<Lagrange polynomial of degree " << q << " with " << n << " points>";
  }

  return s.str();
}
//-----------------------------------------------------------------------------
void Lagrange::init()
{
  // Note: this will be computed each time a new nodal point is specified,
  // but will only be correct once all nodal points are distinct. This
  // requires some extra work at the start but give increased performance
  // since we don't have to check each time that the constants have been
  // computed.

  if (constants == 0)
    constants = new real[n];

  // Compute constants
  for (unsigned int i = 0; i < n; i++)
  {
    real product = 1.0;
    for (unsigned int j = 0; j < n; j++)
      if (j != i)
        product *= points[i] - points[j];
    if (real_abs(product) > real_epsilon())
      constants[i] = 1.0 / product;
  }
}
//-----------------------------------------------------------------------------
