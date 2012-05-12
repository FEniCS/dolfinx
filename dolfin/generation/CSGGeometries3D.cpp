// Copyright (C) 2012 Benjamin Kehlet
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
// First added:  2012-05-12
// Last changed: 2012-05-12

#include "CSGGeometries3D.h"
#include "CSGGeometry.h"
#include "CSGPrimitives3D.h"
#include "CSGOperators.h"

using namespace dolfin;

boost::shared_ptr<CSGGeometry> CSGGeometries::lego( uint n0, uint n1, uint n2, double x0, double x1, double x2 )
{
  // Standard dimensions for LEGO bricks / m
  const double P = 8.0 * 0.001;
  const double h = 3.2 * 0.001;
  const double D = 5.0 * 0.001;
  const double b = 1.7 * 0.001;
  const double d = 0.2 * 0.001;

  // Create brick
  boost::shared_ptr<CSGGeometry> lego(new csg::Box(x0 + 0.5*d, x1 + 0.5*d, x2,
						x0 + n0*P - 0.5*d, x1 + n1*P - 0.5*d, x2 + n2*h));

  // Add knobs
  for (uint i = 0; i < n0; i++)
  {
    for (uint j = 0; j < n1; j++)
    {
      const double x = x0 + (i + 0.5)*P;
      const double y = x1 + (j + 0.5)*P;
      const double z = x2;

      boost::shared_ptr<CSGGeometry> knob(new csg::Cone(Point(x, y, z),
							     Point(x, y, z + n2*h + b),
							     0.5*D, 0.5*D));
      lego = lego + knob;
    }
  }

  return lego;
}
