// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/Quadrature.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Quadrature::Quadrature(unsigned int n)
{
  this->n = n;

  cout << "Allocating for quadrature: n = " << n << endl;
  
  points = new Point[n];
  weights = new real[n];
  
  for (unsigned int i = 0; i < 0; i++)
    weights[i] = 0;

  m = 1.0;
}
//-----------------------------------------------------------------------------
Quadrature::~Quadrature()
{
  if ( points )
    delete [] points;
  points = 0;
  
  if ( weights )
    delete [] weights;
  weights = 0;
}
//-----------------------------------------------------------------------------
int Quadrature::size() const
{
  return n;
}
//-----------------------------------------------------------------------------
const Point& Quadrature::point(unsigned int i) const
{
  dolfin_assert(i < n);
  return points[i];
}
//-----------------------------------------------------------------------------
real Quadrature::weight(unsigned int i) const
{
  dolfin_assert(i < n);
  return weights[i];
}
//-----------------------------------------------------------------------------
real Quadrature::measure() const
{
  return m;
}
//-----------------------------------------------------------------------------
