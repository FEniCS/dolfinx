// Copyright (C) 2012 Anders Logg
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
// Modified by Johannes Ring, 2012
//
// First added:  2012-04-12
// Last changed: 2012-05-03

#include <sstream>
#include <cmath>
#include <dolfin/common/constants.h>
#include <dolfin/math/basic.h>

#include "CSGPrimitives2D.h"
#include "cgal_csg.h"

using namespace dolfin;
using namespace dolfin::csg;

//-----------------------------------------------------------------------------
// Circle
//-----------------------------------------------------------------------------
Circle::Circle(double x0, double x1, double r)
  : _x0(x0), _x1(x1), _r(r)
{
  if (near(_r, 0.0))
  {
    dolfin_error("CSGPrimitives2D.cpp",
                 "create circle",
                 "The radius provided should be greater than zero");
  }
}
//-----------------------------------------------------------------------------
std::string Circle::str(bool verbose) const
{
  std::stringstream s;

  if (verbose)
  {
    s << "<Circle at (" << _x0 << ", " << _x1 << ") with radius " << _r << ">";
  }
  else
  {
    s << "Circle("
      << _x0 << ", " << _x1 << ", " << _r << ")";
  }

  return s.str();
}
//-----------------------------------------------------------------------------
#ifdef HAS_CGAL
csg::Nef_polyhedron_2 Circle::get_cgal_type_2D() const
{
  std::vector<Nef_point_2> points;

  int f = 50; // FIXME: Where should we set the number of fragments?
  //int f = (int) ceil(fmax(fmin(360.0 / 12, _r*DOLFIN_PI), 5));
  for (int i=0; i<f; i++)
  {
    double phi = (2*DOLFIN_PI*i) / f;
    double x, y;
    if (_r > 0) {
      x = _x0 + _r*cos(phi);
      y = _x1 + _r*sin(phi);
    } else {
      x=0;
      y=0;
    }
    points.push_back(Nef_point_2(x, y));
  }

  return Nef_polyhedron_2(points.begin(), points.end(),
			  Nef_polyhedron_2::INCLUDED);
}
#endif
//-----------------------------------------------------------------------------
// Rectangle
//-----------------------------------------------------------------------------
Rectangle::Rectangle(double x0, double x1, double y0, double y1)
  : _x0(x0), _x1(x1), _y0(y0), _y1(y1)
{
  if (near(x0, y0) || near(x1, y1))
  {
    dolfin_error("CSGPrimitives2D.cpp",
                 "create rectangle",
                 "Rectangle with corner (%f, %f) and (%f, %f) degenerated",
		 x0, x1, y0, y1);
  }
}
//-----------------------------------------------------------------------------
std::string Rectangle::str(bool verbose) const
{
  std::stringstream s;

  if (verbose)
  {
    s << "<Rectangle with first corner at (" << _x0 << ", " << _x1 << ") "
      << "and second corner at (" << _y0 << ", " << _y1 << ")>";
  }
  else
  {
    s << "Rectangle("
      << _x0 << ", " << _x1 << ", " << _y0 << ", " << _y1 << ")";
  }

  return s.str();
}
//-----------------------------------------------------------------------------
#ifdef HAS_CGAL
csg::Nef_polyhedron_2 Rectangle::get_cgal_type_2D() const
{
  const double x0 = std::min(_x0, _y0);
  const double y0 = std::max(_x0, _y0);

  const double x1 = std::min(_x1, _y1);
  const double y1 = std::max(_x1, _y1);

  std::vector<Nef_point_2> points;
  points.push_back(Nef_point_2(x0, x1));
  points.push_back(Nef_point_2(y0, x1));
  points.push_back(Nef_point_2(y0, y1));
  points.push_back(Nef_point_2(x0, y1));

  return Nef_polyhedron_2(points.begin(), points.end(),
			  Nef_polyhedron_2::INCLUDED);
}
#endif
//-----------------------------------------------------------------------------
