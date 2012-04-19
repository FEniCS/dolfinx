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
// Last changed: 2012-04-19

#include <sstream>

#include "CSGPrimitives3D.h"

using namespace dolfin;
using namespace dolfin::csg;

//-----------------------------------------------------------------------------
// Sphere
//-----------------------------------------------------------------------------
Sphere::Sphere(double x0, double x1, double x2, double r)
  : _x0(x0), _x1(x1), _x2(x2), _r(r)
{
  // FIXME: Check validity of coordinates here
}
//-----------------------------------------------------------------------------
std::string Sphere::str(bool verbose) const
{
  std::stringstream s;

  if (verbose)
  {
    s << "<Sphere at (" << _x0 << ", " << _x1 << ", " << _x2 << ") "
      << "with radius " << _r << ">";
  }
  else
  {
    s << "Sphere("
      << _x0 << ", " << _x1 << ", " << _x2 << ", " << _r << ")";
  }

  return s.str();
}
//-----------------------------------------------------------------------------
#ifdef HAS_CGAL
Nef_polyhedron_3 Sphere::get_cgal_type_3D() const
{
  //FIXME
  return Nef_polyhedron_3();
}
#endif    
//-----------------------------------------------------------------------------
// Box
//-----------------------------------------------------------------------------
Box::Box(double x0, double x1, double x2,
         double y0, double y1, double y2)
  : _x0(x0), _x1(x1), _x2(x2), _y0(y0), _y1(y1), _y2(y2)
{
  // FIXME: Check validity of coordinates here
}
//-----------------------------------------------------------------------------
std::string Box::str(bool verbose) const
{
  std::stringstream s;

  if (verbose)
  {
    s << "<Box with first corner at (" << _x0 << ", " << _x1 << ", " << _x2 << ") "
      << "and second corner at (" << _x0 << ", " << _x1 << ", " << _x2 << ")>";
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
#ifdef HAS_CGAL
Nef_polyhedron_3 Box::get_cgal_type_3D() const
{
  // FIXME
  return Nef_polyhedron_3();
}
#endif    
