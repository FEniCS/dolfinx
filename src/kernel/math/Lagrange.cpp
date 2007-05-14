// Copyright (C) 2003-2005 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2006.
//
// First added:  2003-06-12
// Last changed: 2006-03-27

#include <cmath>
#include <dolfin/dolfin_log.h>
#include <dolfin/Lagrange.h>

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
  dolfin_assert(i <= q);

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
  dolfin_assert(i <= q);

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
  dolfin_assert(i <= q);

  real product(constants[i]);
  for (unsigned int j = 0; j < n; j++)
    if ( j != i )
      product *= x - points[j];
  
  return product;
}
//-----------------------------------------------------------------------------
real Lagrange::ddx(unsigned int i, real x)
{
  dolfin_assert(i <= q);
  
  real sum(0);
  for (unsigned int j = 0; j < n; j++) {
    if ( j != i ) {
      real product = 1.0;
      for (unsigned int k = 0; k < n; k++)
	if ( k != i && k != j )
	  product *= x - points[k];
      sum += product;
    }
  }

  return sum * constants[i];
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
LogStream& dolfin::operator<<(LogStream& stream, const Lagrange& p)
{
  stream << "[ Lagrange polynomial of degree " << p.q << " with " << p.n << " points ]";
  return stream;
}
//-----------------------------------------------------------------------------
void Lagrange::disp() const
{
  message("Lagrange polynomial of degree %d with %d points.", q, n);
  message("----------------------------------------------");

  for (unsigned int i = 0; i < n; i++)
    message("x[%d] = %f", i, points[i]);
}
//-----------------------------------------------------------------------------
void Lagrange::init()
{
  // Note: this will be computed each time a new nodal point is specified,
  // but will only be correct once all nodal points are distinct. This
  // requires some extra work at the start but give increased performance
  // since we don't have to check each time that the constants have been
  // computed.

  if ( constants == 0 )
    constants = new real[n];

  // Compute constants
  for (unsigned int i = 0; i < n; i++) {
    real product = 1.0;
    for (unsigned int j = 0; j < n; j++)
      if ( j != i )
	product *= points[i] - points[j];
    if ( fabs(product) > DOLFIN_EPS )
      constants[i] = 1.0 / product;
  }
}
//-----------------------------------------------------------------------------
