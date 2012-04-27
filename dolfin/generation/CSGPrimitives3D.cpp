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
// Last changed: 2012-04-28

#include <sstream>
#include <dolfin/math/basic.h>
#include <dolfin/log/LogStream.h>
#include "CSGPrimitives3D.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
// Sphere
//-----------------------------------------------------------------------------
csg::Sphere::Sphere(double x0, double x1, double x2, double r)
  : _x0(x0), _x1(x1), _x2(x2), _r(r)
{
  // FIXME: Check validity of coordinates here
}
//-----------------------------------------------------------------------------
std::string csg::Sphere::str(bool verbose) const
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
csg::Nef_polyhedron_3 csg::Sphere::get_cgal_type_3D() const
{
  //FIXME
  return Nef_polyhedron_3();
}
#endif    
//-----------------------------------------------------------------------------
// Box
//-----------------------------------------------------------------------------
csg::Box::Box(double x0, double x1, double x2,
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
std::string csg::Box::str(bool verbose) const
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
#ifdef HAS_CGAL
csg::Nef_polyhedron_3 csg::Box::get_cgal_type_3D() const
{
  typedef typename Exact_Polyhedron_3::Halfedge_handle Halfedge_handle;

  const double x0 = std::min(_x0, _y0);
  const double y0 = std::max(_x0, _y0);

  const double x1 = std::min(_x1, _y1);
  const double y1 = std::max(_x1, _y1);

  const double x2 = std::min(_x2, _y2);
  const double y2 = std::max(_x2, _y2);

  Point_3 p0(y0,   x1,  x2);
  Point_3 p1( x0,  x1,  y2);
  Point_3 p2( x0,  x1,  x2);
  Point_3 p3( x0,  y1,  x2);
  Point_3 p4( y0,  x1,  y2);
  Point_3 p5( x0,  y1,  y2);
  Point_3 p6( y0,  y1,  x2);
  Point_3 p7( y0,  y1,  y2);
  
  Exact_Polyhedron_3 P;
  Halfedge_handle h = P.make_tetrahedron( p0, p1, p2, p3);

  Halfedge_handle g = h->next()->opposite()->next();
  P.split_edge( h->next());
  P.split_edge( g->next());
  P.split_edge( g);
  h->next()->vertex()->point()     = p4;
  g->next()->vertex()->point()     = p5;
  g->opposite()->vertex()->point() = p6;
  Halfedge_handle f = P.split_facet( g->next(),
				     g->next()->next()->next());
  Halfedge_handle e = P.split_edge( f);
  e->vertex()->point() = p7;
  P.split_facet( e, f->next()->next());

  return  csg::Nef_polyhedron_3(P);;
}
#endif    
