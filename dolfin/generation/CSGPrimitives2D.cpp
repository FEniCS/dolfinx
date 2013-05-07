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
// Modified by Benjamin Kehlet, 2012-2013
//
// First added:  2012-04-12
// Last changed: 2013-03-13

#include <sstream>
#include <dolfin/math/basic.h>

#include "CSGPrimitives2D.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
// Circle
//-----------------------------------------------------------------------------
Circle::Circle(double x0, double x1, double r, std::size_t fragments)
  : _x0(x0), _x1(x1), _r(r), _fragments(fragments)
{
  if (_r < DOLFIN_EPS)
  {
    dolfin_error("CSGPrimitives2D.cpp",
                 "create circle",
                 "Circle with center (%f, %f) has zero or negative radius",
                 _x0, _x1);
  }
  if (_fragments < 3)
  {
    dolfin_error("CSGPrimitives2D.cpp",
                 "create circle",
                 "Unable to create circle with less than 3 fragments");

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
// Ellipse
//-----------------------------------------------------------------------------
Ellipse::Ellipse(double x0, double x1, double a, double b,
                 std::size_t fragments)
  : _x0(x0), _x1(x1), _a(a), _b(b), _fragments(fragments)
{
  if (_a < DOLFIN_EPS || _b < DOLFIN_EPS)
  {
    dolfin_error("CSGPrimitives2D.cpp",
                 "create ellipse",
                 "Ellipse with center (%f, %f) has invalid semi-axis",
                 _x0, _x1);
  }
  if (_fragments < 3)
  {
    dolfin_error("CSGPrimitives2D.cpp",
                 "create ellipse",
                 "Unable to create ellipse with less than 3 fragments");

  }
}
//-----------------------------------------------------------------------------
std::string Ellipse::str(bool verbose) const
{
  std::stringstream s;

  if (verbose)
  {
    s << "<Ellipse at (" << _x0 << ", " << _x1 << ") with horizontal semi-axis "
      << _a << " and vertical semi-axis " << _b << ">";
  }
  else
  {
    s << "Ellipse("
      << _x0 << ", " << _x1 << ", " << _a << ", " << _b << ")";
  }

  return s.str();
}
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
// Polygon
//-----------------------------------------------------------------------------
Polygon::Polygon(const std::vector<Point>& vertices)
  : _vertices(vertices.begin(), vertices.end())
{
  if (_vertices.size() < 3)
  {
    dolfin_error("CSGPrimitives2D.cpp",
                 "create polygon",
                 "Polygon should have at least three vertices");
  }
}
//-----------------------------------------------------------------------------
std::string Polygon::str(bool verbose) const
{
  std::stringstream s;

  if (verbose)
  {
    s << "<Polygon with vertices ";
    std::vector<Point>::const_iterator p;
    for (p = _vertices.begin(); p != _vertices.end(); ++p)
    {
      s << "(" << p->x() << ", " << p->y() << ")";
      if ((p != _vertices.end()) && (p + 1 != _vertices.end()))
        s << ", ";
    }
    s << ">";
  }
  else
  {
    s << "Polygon(";
    std::vector<Point>::const_iterator p;
    for (p = _vertices.begin(); p != _vertices.end(); ++p)
    {
      s << "(" << p->x() << ", " << p->y() << ")";
      if ((p != _vertices.end()) && (p + 1 != _vertices.end()))
        s << ", ";
    }
    s << ")";
  }

  return s.str();
}
//-----------------------------------------------------------------------------
