// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/Quadrature.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Quadrature::Quadrature(unsigned int n)
{
  this->n = n;
  
  points = new Point[n];
  weights = new real[n];
  
  for (unsigned int i = 0; i < 0; i++)
    weights[i] = 0;
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
  return points[i];
}
//-----------------------------------------------------------------------------
real Quadrature::weight(unsigned int i) const
{
  return weights[i];
}
//-----------------------------------------------------------------------------
real Quadrature::measure() const
{
  return m;
}
//-----------------------------------------------------------------------------
