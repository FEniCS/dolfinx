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
// Convenience routine to make debugging easier. Remove before releasing.
static void add_facet(CGAL::Polyhedron_incremental_builder_3<csg::Exact_HalfedgeDS>& builder, 
		      std::vector<int>& vertices, bool print=false)
{
  static int facet_no = 0;

  if (print)
  {
    cout << "Begin facet " << facet_no << endl;
    if (!vertices.size())
    {
      cout << "No vertices in facet!" << endl;
      return;
    }

    // Print vertices
    for (std::vector<int>::iterator it=vertices.begin(); it != vertices.end(); it++)
    {
      cout << "Vertex: " << (*it) << endl;
    }

    if (builder.test_facet(vertices.begin(), vertices.end()))
      cout << "Facet ok, size: " << vertices.size() << endl;
    else
      cout << "Facet not ok" << endl;
  }

  builder.begin_facet();
  for (std::vector<int>::iterator it=vertices.begin(); it != vertices.end(); it++)
  {
    builder.add_vertex_to_facet(*it);
  }
  builder.end_facet();

  if (print)
    cout << "End facet" << endl;
  facet_no++;
}
//-----------------------------------------------------------------------------
static void add_vertex(CGAL::Polyhedron_incremental_builder_3<csg::Exact_HalfedgeDS>& builder, 
		       const csg::Point_3& point, bool print=false)
{
  static int vertex_no = 0;
  if (print) 
  {
    std::cout << "Adding vertex " << vertex_no << " at " << point << std::endl;
  }

  builder.add_vertex(point);
  vertex_no++;
}
//-----------------------------------------------------------------------------
// Sphere
//-----------------------------------------------------------------------------
csg::Sphere::Sphere(Point c, double r, uint slices)
  : c(c), r(r), slices(slices)
{
  // FIXME: Check validity of coordinates here
}
//-----------------------------------------------------------------------------
std::string csg::Sphere::str(bool verbose) const
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
#ifdef HAS_CGAL
class Build_sphere : public CGAL::Modifier_base<csg::Exact_HalfedgeDS> 
{
 public:
  Build_sphere(const csg::Sphere& sphere) : sphere(sphere){}

  void operator()( csg::Exact_HalfedgeDS& hds )
  {
    // FIXME
    const uint slices = 0;
    const uint num_sectors = sphere.slices;

    const dolfin::Point top = sphere.c + Point(sphere.r, 0, 0);
    const dolfin::Point bottom = sphere.c - Point(sphere.r, 0, 0);
    const dolfin::Point axis = Point(1, 0, 0);

    //const int 

    CGAL::Polyhedron_incremental_builder_3<csg::Exact_HalfedgeDS> builder( hds, true);

    builder.begin_surface(slices+5, slices + 10);

    for (uint i = 0; i < num_sectors; i++)
    {
      const Point direction = Point(0, 1, 0).rotate(axis, i*2.0*DOLFIN_PI/num_sectors);
      const Point v = sphere.c + direction*sphere.r;
      add_vertex(builder, csg::Point_3 (v.x(), v.y(), v.z()));
    }

    // Add top and bottom vertex
    add_vertex(builder, csg::Point_3(top.x(), top.y(), top.z()));
    add_vertex(builder, csg::Point_3(bottom.x(), bottom.y(), bottom.z()));


    // Add the top and bottom facets
    for (uint i = 0; i < num_sectors; i++)
    {
      {
	// Top facet
	std::vector<int> f;
	f.push_back( num_sectors );
	f.push_back( i );
	f.push_back( (i+1)%num_sectors );
	add_facet(builder, f);
      }

      {
      	// Bottom facet
      	std::vector<int> f;
      	//const int offset = 0;
      	f.push_back( num_sectors+1 );
      	f.push_back( (i+1) % num_sectors );
      	f.push_back( i);
      	add_facet(builder, f);
      }
    }

    builder.end_surface();

  }

  private:
  const csg::Sphere& sphere;
};
//-----------------------------------------------------------------------------
csg::Nef_polyhedron_3 csg::Sphere::get_cgal_type_3D() const
{
  Exact_Polyhedron_3 P;
  Build_sphere builder(*this);
  P.delegate(builder);
  dolfin_assert(P.is_valid());
  dolfin_assert(P.is_closed());
  return csg::Nef_polyhedron_3(P);
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

  return csg::Nef_polyhedron_3(P);;
}
#endif
//-----------------------------------------------------------------------------
// Cone
//-----------------------------------------------------------------------------
csg::Cone::Cone(Point top, Point bottom, double top_radius, double bottom_radius, dolfin::uint slices)
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
std::string csg::Cone::str(bool verbose) const
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
#ifdef HAS_CGAL
// Return some vector orthogonal to a
static Point generate_orthogonal(const Point& a)
{
  const Point b(0, 1, 0);
  const Point c(0, 0, 1);
  
  // Find a vector not parallel to a.
  const Point d = (fabs(a.dot(b)) < fabs(a.dot(c))) ? b : c;
  return a.cross(d);
}
//-----------------------------------------------------------------------------
class Build_cone : public CGAL::Modifier_base<csg::Exact_HalfedgeDS> 
{
 public:
  Build_cone(const csg::Cone& cone) : cone(cone){}

  void operator()( csg::Exact_HalfedgeDS& hds )
  {
    const dolfin::Point axis = cone.top - cone.bottom;
    dolfin::Point initial = generate_orthogonal(axis/axis.norm());

    CGAL::Polyhedron_incremental_builder_3<csg::Exact_HalfedgeDS> builder( hds, true);

    const int num_faces = cone.slices;
    const bool top_degenerate = near(cone.top_radius, 0.0);
    const bool bottom_degenerate = near(cone.bottom_radius, 0.0);

    const int num_vertices = (top_degenerate || bottom_degenerate) ? num_faces+1 : num_faces*2;

    builder.begin_surface(num_vertices, num_faces + 2);

    if (top_degenerate) 
    {
      // A single vertex at the top.
      const csg::Point_3 p(cone.top.x(), cone.top.y(), cone.top.z());
      builder.add_vertex(p);
    }

    if (bottom_degenerate) 
    {
      // A single vertex at the bottom.
      const csg::Point_3 p(cone.bottom.x(), cone.bottom.y(), cone.bottom.z());
      builder.add_vertex(p);
    }

    const double delta_theta = 2.0 * DOLFIN_PI / num_faces;
    for (int i = 0; i < num_faces; ++i) 
    {
      const double theta = i*delta_theta;
      const Point rotated = initial.rotate(axis, theta);

      if (!bottom_degenerate)
      {
	const Point p = cone.bottom + rotated*cone.bottom_radius;
	const csg::Point_3 p_(p.x(), p.y(), p.z());
	builder.add_vertex(p_);
      }

      if (!top_degenerate) 
      {
	const Point p = cone.top + rotated*cone.top_radius;
	const csg::Point_3 p_(p.x(), p.y(), p.z());
        builder.add_vertex(p_);
      }
    }

    // Construct the facets on the side. 
    // Vertices must be sorted counter clockwise seen from inside.
    for (int i = 0; i < num_faces; ++i) 
    {
      if (top_degenerate) 
      {
	std::vector<int> f;
	f.push_back(0);
	f.push_back(i+1);
	f.push_back((i+1)%num_faces + 1);
	add_facet(builder, f);
      } else if (bottom_degenerate) 
      {
	std::vector<int> f;
	f.push_back(0);
	f.push_back((i + 1) % num_faces + 1);
	f.push_back(i+1);
	add_facet(builder, f);
      } else 
      {
	// NOTE: Had to draw the sides with two triangles,
	// instead of quads. Couldn't get CGAL to accept
	// the quads. Don't know if it as a problem here
	// or on the CGAL side. BK

	const int vertex_to_add = i*2;

	// First triangle
	std::vector<int> f;
	f.push_back(vertex_to_add+1);
	f.push_back(vertex_to_add);
	f.push_back((vertex_to_add + 2) % num_vertices);
	add_facet(builder, f);

	// Second triangle
	std::vector<int> g;
	g.push_back((vertex_to_add + 2) % num_vertices);
	g.push_back((vertex_to_add + 3) % num_vertices);
	g.push_back(vertex_to_add+1);
	add_facet(builder, g);
      }
    }

    // Construct the the top facet
    if (!top_degenerate) 
    {
      std::vector<int> f;      
      for (int i = 0; i < num_faces; i++)
      {
	f.push_back(bottom_degenerate ? i+1 : i*2 +1 );
      }
      add_facet(builder, f);
    }


    // Construct the bottom facet.
    if (!bottom_degenerate) 
    {
      std::vector<int> f;
      for (int i = num_faces-1; i >= 0; i -= 1) 
      {
	f.push_back(top_degenerate ? i+1 : i*2);
      }
      add_facet(builder, f);
    }

    builder.end_surface();
  }
private:
  const csg::Cone& cone;
};
//-----------------------------------------------------------------------------
csg::Nef_polyhedron_3 csg::Cone::get_cgal_type_3D() const
{
  Exact_Polyhedron_3 P;
  Build_cone builder(*this);
  P.delegate(builder);
  dolfin_assert(P.is_closed());
  return csg::Nef_polyhedron_3(P);
}
#endif    
