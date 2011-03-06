// Copyright (C) 2003-2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells
//
// First added:  2003-02-06
// Last changed: 2006-10-23

#include <dolfin/log/dolfin_log.h>
#include "Quadrature.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
Quadrature::Quadrature(unsigned int n, real m) : points(n), weights(n, 0), m(m)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Quadrature::~Quadrature()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
int Quadrature::size() const
{
  return points.size();
}
//-----------------------------------------------------------------------------
real Quadrature::point(unsigned int i) const
{
  assert(i < points.size());
  return points[i];
}
//-----------------------------------------------------------------------------
real Quadrature::weight(unsigned int i) const
{
  assert(i < points.size());
  return weights[i];
}
//-----------------------------------------------------------------------------
real Quadrature::measure() const
{
  return m;
}
//-----------------------------------------------------------------------------
