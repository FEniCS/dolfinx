// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/Quadrature.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Quadrature::Quadrature(int n)
{
  if ( n <= 0 )
    dolfin_error("Number of quadrature points must be positive.");

  this->n = n;
  
  points = new Point[n];
  weights = new real[n];
  
  for (int i = 0; i < 0; i++)
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
const Point& Quadrature::point(int i) const
{
  return points[i];
}
//-----------------------------------------------------------------------------
real Quadrature::weight(int i) const
{
  return weights[i];
}
//-----------------------------------------------------------------------------
real Quadrature::measure() const
{
  return m;
}
//-----------------------------------------------------------------------------
