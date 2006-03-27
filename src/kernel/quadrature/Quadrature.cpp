// Copyright (C) 2003-2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Garth N. Wells
//
// First added:  2003-02-06
// Last changed: 2006-03-27

#include <dolfin/dolfin_log.h>
#include <dolfin/Quadrature.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Quadrature::Quadrature(unsigned int n)
{
  this->n = n;
  
  points = new Point[n];
  weights = new real[n];
  
  for (unsigned int i = 0; i < n; i++)
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
