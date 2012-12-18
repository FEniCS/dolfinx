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
// Modified by Benjamin Kehlet, 2012
//
// First added:  2012-04-12
// Last changed: 2012-11-12

#include <sstream>
#include <dolfin/math/basic.h>
#include <dolfin/log/LogStream.h>
#include "CSGPrimitives3D.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
// Sphere
//-----------------------------------------------------------------------------
Sphere::Sphere(Point c, double r, std::size_t slices)
  : c(c), r(r), slices(slices)
{
  if (r < DOLFIN_EPS)
    dolfin_error("CSGPrimitives3D.cpp",
		   "Create sphere",
		   "Sphere with center (%f, %f, %f) has zero or negative radius", c.x(), c.y(), c.z());

  if (slices < 1)
  {
    dolfin_error("CSGPrimitives3D.cpp",
		 "Create sphere",
		 "Can't create sphere with zero slices");

  }
}
//-----------------------------------------------------------------------------
std::string Sphere::str(bool verbose) const
{
  std::stringstream s;

  if (verbose)
  {
    s << "<Sphere at " << c << " "
      << "with radius " << r << ">";
  }
  else
  {
    s << "Sphere(" << c << ", " << r << ")";
  }

  return s.str();
}
//-----------------------------------------------------------------------------
// Box
//-----------------------------------------------------------------------------
Box::Box(double x0, double x1, double x2,
         double y0, double y1, double y2)
  : _x0(x0), _x1(x1), _x2(x2), _y0(y0), _y1(y1), _y2(y2)
{
  // FIXME: Check validity of coordinates here
  if (near(x0, y0) || near(x1, y2) || near(x2, y2))
      dolfin_error("CSGPrimitives3D.cpp",
		   "Create axis aligned box",
		   "Box with corner (%f, %f, %f) and (%f, %f, %f) degenerated", x0, x1, x2, y0, y1, y2);
}
//-----------------------------------------------------------------------------
std::string Box::str(bool verbose) const
{
  std::stringstream s;

  if (verbose)
  {
    s << "<Box with first corner at (" << _x0 << ", " << _x1 << ", " << _x2 << ") "
      << "and second corner at (" << _y0 << ", " << _y1 << ", " << _y2 << ")>";
  }
  else
  {
    s << "Box("
      << _x0 << ", " << _x1 << ", " << _x2 << ", "
      << _y0 << ", " << _y1 << ", " << _y2 << ")";
  }

  return s.str();
}
//-----------------------------------------------------------------------------
// Cone
//-----------------------------------------------------------------------------
Cone::Cone(Point top, Point bottom, double top_radius, double bottom_radius, std::size_t slices)
  : top(top), bottom(bottom), top_radius(top_radius), bottom_radius(bottom_radius), slices(slices)
{
  if (near(top_radius, 0.0) && near(bottom_radius, 0.0))
      dolfin_error("CSGPrimitives3D.cpp",
		   "Create cone",
		   "Cone with zero thickness");

  if (top.distance(bottom) < DOLFIN_EPS)
    dolfin_error("CSGPrimitives3D.cpp",
		 "Create cone",
		 "Cone with zero length");

}
//-----------------------------------------------------------------------------
std::string Cone::str(bool verbose) const
{
  std::stringstream s;

  if (verbose)
  {
    s << "<Cone with top at " << top << ", top radius " << top_radius
      << " and bottom at " << bottom << ", bottom radius " << bottom_radius << ", with " << slices << " slices>";
  }
  else
  {
    s << "Cone( "
      << top << ", " << bottom << ", " << top_radius << ", " << bottom_radius << " )";
  }

  return s.str();
}
//-----------------------------------------------------------------------------
Tetrahedron::Tetrahedron(Point x0, Point x1, Point x2, Point x3)
  : x0(x0), x1(x1), x2(x2), x3(x3)
{}
//-----------------------------------------------------------------------------
/// Informal string representation
std::string Tetrahedron::str(bool verbose) const
{
  std::stringstream s;

  if (verbose)
  {
    s << "<Tetrahedron with point at " << x0 << ", " << x1 << ", " << x2 << ", " << x3 << ">";
  }
  else
  {
    s << "Tetrahedron( " << x0 << ", " << x1 << ", " << x2 << ", " << x3 << ")";

  }

  return s.str();
}
//-----------------------------------------------------------------------------
Surface3D::Surface3D(std::string filename)
  : filename(filename)
{}
//-----------------------------------------------------------------------------
std::string Surface3D::str(bool verbose) const
{
  return std::string("Surface3D from file ") + filename;
}
