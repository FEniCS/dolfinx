// Copyright (C) 2003-2006 Anders Logg
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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Garth N. Wells
//
// First added:  2003-02-06
// Last changed: 2006-10-23

#include <dolfin/log/dolfin_log.h>
#include "Quadrature.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
Quadrature::Quadrature(unsigned int n, double m) : points(n), weights(n, 0), m(m)
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
double Quadrature::point(unsigned int i) const
{
  dolfin_assert(i < points.size());
  return points[i];
}
//-----------------------------------------------------------------------------
double Quadrature::weight(unsigned int i) const
{
  dolfin_assert(i < points.size());
  return weights[i];
}
//-----------------------------------------------------------------------------
double Quadrature::measure() const
{
  return m;
}
//-----------------------------------------------------------------------------
