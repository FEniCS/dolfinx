// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <cmath>
#include <dolfin/dolfin_log.h>
#include <dolfin/Lagrange.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Lagrange::Lagrange(int q)
{
  if ( q < 0 )
    dolfin_error("Degree for Lagrange polynomial must be non-negative.");

  this->q = q;
  n = q + 1;
  
  points = new real[n];
  for (int i = 0; i < n; i++)
    points[i] = 0.0;

  constants = 0;
  updated = false;
}
//-----------------------------------------------------------------------------
Lagrange::Lagrange(const Lagrange& p)
{
  dolfin_assert(p.q >= 0);

  this->q = p.q;
  n = p.n;

  points = new real[p.n];
  for (int i = 0; i < p.n; i++)
    points[i] = p.points[i];

  constants = 0;
  updated = false;
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
void Lagrange::set(int i, real x)
{
  dolfin_assert(i >= 0);
  dolfin_assert(i <= q);

  points[i] = x;
  updated = false;
}
//-----------------------------------------------------------------------------
int Lagrange::size() const
{
  return n;
}
//-----------------------------------------------------------------------------
int Lagrange::degree() const
{
  return q;
}
//-----------------------------------------------------------------------------
real Lagrange::point(int i) const
{
  dolfin_assert(i >= 0);
  dolfin_assert(i <= q);

  return points[i];
}
//-----------------------------------------------------------------------------
real Lagrange::operator() (int i, real x)
{
  return eval(i,x);
}
//-----------------------------------------------------------------------------
real Lagrange::eval(int i, real x)
{
  dolfin_assert(i >= 0);
  dolfin_assert(i <= q);

  init();

  double product = constants[i];
  
  for (int j = 0; j < n; j++)
    if ( j != i )
      product *= x - points[j];
  
  return product;
}
//-----------------------------------------------------------------------------
real Lagrange::dx(int i, real x)
{
  dolfin_assert(i >= 0);
  dolfin_assert(i <= q);
  
  init();
  
  real sum = 0.0;
  for (int j = 0; j < n; j++) {
    if ( j != i ) {
      real product = 1.0;
      for (int k = 0; k < n; k++)
	if ( k != i && k != j )
	  product *= x - points[k];
      sum += product;
    }
  }

  return sum * constants[i];
}
//-----------------------------------------------------------------------------
real Lagrange::dqx(int i)
{
  dolfin_assert(i >= 0);

  init();
  
  real product = constants[i];
  
  for (int j = 1; j <= q; j++)
    product *= (double) j;
  
  return product;
}
//-----------------------------------------------------------------------------
LogStream& dolfin::operator<<(LogStream& stream, const Lagrange& p)
{
  stream << "[ Lagrange polynomial of degree " << p.q << " with " << p.n << " points ]";
  return stream;
}
//-----------------------------------------------------------------------------
void Lagrange::show() const
{
  dolfin_info("Lagrange polynomial of degree %d with %d points.", q, n);
  dolfin_info("----------------------------------------------");

  for (int i = 0; i < n; i++)
    dolfin_info("x[%d] = %f", i, points[i]);
}
//-----------------------------------------------------------------------------
void Lagrange::init()
{
  if ( updated )
    return;

  if ( constants == 0 )
    constants = new real[n];

  // Compute constants
  for (int i = 0; i < n; i++) {
    real product = 1.0;
    for (int j = 0; j < n; j++)
      if ( j != i )
	product *= points[i] - points[j];
    if ( fabs(product) < DOLFIN_EPS )
      dolfin_error("Nodal points for Lagrange polynomial must be distinct.");
    constants[i] = 1.0 / product;
  }
  
  updated = true;
}
//-----------------------------------------------------------------------------
