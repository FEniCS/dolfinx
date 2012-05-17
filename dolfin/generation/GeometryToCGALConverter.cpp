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
// First added:  2012-05-10
// Last changed: 2012-05-10

// This class is capable of converting a 3D dolfin::CSGGeometry to a 
// CGAL::Polyhedron_3

#include "GeometryToCGALConverter.h"
#include "CSGGeometry.h"
#include "CSGOperators.h"
#include "CSGPrimitives3D.h"
#include <dolfin/mesh/Point.h>
#include <dolfin/log/LogStream.h>
#include <dolfin/math/basic.h>

#include "cgal_csg3d.h"

#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Nef_polyhedron_3.h>
#include <CGAL/Polyhedron_incremental_builder_3.h>

#include <CGAL/Polyhedron_incremental_builder_3.h>

typedef CGAL::Exact_predicates_exact_constructions_kernel Exact_Kernel;
typedef CGAL::Nef_polyhedron_3<Exact_Kernel> Nef_polyhedron_3;
typedef CGAL::Polyhedron_3<Exact_Kernel> Exact_Polyhedron_3;
typedef Exact_Polyhedron_3::HalfedgeDS Exact_HalfedgeDS;
typedef Nef_polyhedron_3::Point_3 Exact_Point_3;
// typedef Nef_polyhedron_3::Plane_3 Plane_3;

using namespace dolfin;

//-----------------------------------------------------------------------------
// Taken from demo/Polyhedron/Scene_nef_polyhedron_item.cpp in the
// CGAL source tree.
// Quick hacks to convert polyhedra from exact to inexact and
// vice-versa
template <class Polyhedron_input, class Polyhedron_output>
struct Copy_polyhedron_to
  : public CGAL::Modifier_base<typename Polyhedron_output::HalfedgeDS>
{
  Copy_polyhedron_to(const Polyhedron_input& in_poly)
    : in_poly(in_poly) {}

  void operator()(typename Polyhedron_output::HalfedgeDS& out_hds)
  {
    typedef typename Polyhedron_output::HalfedgeDS Output_HDS;
    typedef typename Polyhedron_input::HalfedgeDS Input_HDS;

    CGAL::Polyhedron_incremental_builder_3<Output_HDS> builder(out_hds);

    typedef typename Polyhedron_input::Vertex_const_iterator Vertex_const_iterator;
    typedef typename Polyhedron_input::Facet_const_iterator  Facet_const_iterator;
    typedef typename Polyhedron_input::Halfedge_around_facet_const_circulator HFCC;

    builder.begin_surface(in_poly.size_of_vertices(),
      in_poly.size_of_facets(),
      in_poly.size_of_halfedges());

    for(Vertex_const_iterator
      vi = in_poly.vertices_begin(), end = in_poly.vertices_end();
      vi != end ; ++vi)
    {
      typename Polyhedron_output::Point_3 p(::CGAL::to_double( vi->point().x()),
	::CGAL::to_double( vi->point().y()),
	::CGAL::to_double( vi->point().z()));
      builder.add_vertex(p);
    }

    typedef CGAL::Inverse_index<Vertex_const_iterator> Index;
    Index index( in_poly.vertices_begin(), in_poly.vertices_end());

    for(Facet_const_iterator
      fi = in_poly.facets_begin(), end = in_poly.facets_end();
      fi != end; ++fi)
    {
      HFCC hc = fi->facet_begin();
      HFCC hc_end = hc;
      //     std::size_t n = circulator_size( hc);
      //     CGAL_assertion( n >= 3);
      builder.begin_facet ();
      do {
	builder.add_vertex_to_facet(index[hc->vertex()]);
	++hc;
      } while( hc != hc_end);
      builder.end_facet();
    }
    builder.end_surface();
  } // end operator()(..)
private:
  const Polyhedron_input& in_poly;
}; // end Copy_polyhedron_to<>

template <class Poly_A, class Poly_B>
void copy_to(const Poly_A& poly_a, Poly_B& poly_b)
{
  Copy_polyhedron_to<Poly_A, Poly_B> modifier(poly_a);
  poly_b.delegate(modifier);
  CGAL_assertion(poly_b.is_valid());
}
//-----------------------------------------------------------------------------

// // Convenience routine to make debugging easier. Remove before releasing.
static void add_facet(CGAL::Polyhedron_incremental_builder_3<Exact_HalfedgeDS>& builder, 
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
static void add_vertex(CGAL::Polyhedron_incremental_builder_3<Exact_HalfedgeDS>& builder, 
		       const Exact_Point_3& point, bool print=false)
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
class Build_sphere : public CGAL::Modifier_base<Exact_HalfedgeDS> 
{
 public:
  Build_sphere(const csg::Sphere& sphere) : sphere(sphere){}

  void operator()( Exact_HalfedgeDS& hds )
  {
    const dolfin::uint num_slices = sphere.slices;
    const dolfin::uint num_sectors = (sphere.slices+1) * 2;

    const dolfin::Point top = sphere.c + Point(sphere.r, 0, 0);
    const dolfin::Point bottom = sphere.c - Point(sphere.r, 0, 0);
    const dolfin::Point axis = Point(1, 0, 0);

    const int num_vertices = num_slices*num_sectors+2;
    const int num_facets = num_sectors*2*num_slices;

    CGAL::Polyhedron_incremental_builder_3<Exact_HalfedgeDS> builder( hds, true );

    builder.begin_surface(num_vertices, num_facets);

    const Point slice_rotation_axis(0, 1, 0);

    for (dolfin::uint i = 0; i < num_slices; i++)
    {
      const Point sliced = axis.rotate(slice_rotation_axis, (i+1)*DOLFIN_PI/(num_slices+1));
      for (dolfin::uint j = 0; j < num_sectors; j++)
      {
	const Point direction = sliced.rotate(axis, j*2.0*DOLFIN_PI/num_sectors);
	const Point v = sphere.c + direction*sphere.r;
	add_vertex(builder, Exact_Point_3 (v.x(), v.y(), v.z()));
      }
    }

    // Add top and bottom vertex
    add_vertex(builder, Exact_Point_3(top.x(), top.y(), top.z()));
    add_vertex(builder, Exact_Point_3(bottom.x(), bottom.y(), bottom.z()));


    // Add the side facets
    for (dolfin::uint i = 0; i < num_slices-1; i++)
    {
      for (dolfin::uint j = 0; j < num_sectors; j++)
      {
	const dolfin::uint offset1 = i*num_sectors;
	const dolfin::uint offset2 = (i+1)*num_sectors;

	{
	  std::vector<int> f;
	  f.push_back(offset1 + j);
	  f.push_back(offset1 + (j+1)%num_sectors);
	  f.push_back(offset2 + j);
	  add_facet(builder, f);
	}

	{
	  std::vector<int> f;
	  f.push_back(offset2 + (j+1)%num_sectors);
	  f.push_back(offset2 + j);
	  f.push_back(offset1 + (j+1)%num_sectors);
	  add_facet(builder, f);
	}
	
      }
    }

    // Add the top and bottom facets
    const dolfin::uint bottom_offset = num_sectors*(num_slices-1);

    for (dolfin::uint i = 0; i < num_sectors; i++)
    {
      {
	// Top facet
	std::vector<int> f;
	f.push_back( num_vertices-2 );
	f.push_back( (i+1)%num_sectors );
	f.push_back( i );
	add_facet(builder, f);
      }
      
      {
	// Bottom facet
	std::vector<int> f;
	//const int offset = 0;
	f.push_back( num_vertices-1 );
	f.push_back( bottom_offset + i);
	f.push_back( bottom_offset + (i+1)%num_sectors );
	add_facet(builder, f);
      }
    }

    builder.end_surface();

  }

  private:
  const csg::Sphere& sphere;
};
//-----------------------------------------------------------------------------
static Nef_polyhedron_3 make_sphere(const csg::Sphere* s)
{
  Exact_Polyhedron_3 P;
  Build_sphere builder(*s);
  P.delegate(builder);
  dolfin_assert(P.is_valid());
  dolfin_assert(P.is_closed());
  return Nef_polyhedron_3(P);
}
//-----------------------------------------------------------------------------
static Nef_polyhedron_3 make_box(const csg::Box* b)
{
  typedef typename Exact_Polyhedron_3::Halfedge_handle Halfedge_handle;

  const double x0 = std::min(b->_x0, b->_y0);
  const double y0 = std::max(b->_x0, b->_y0);

  const double x1 = std::min(b->_x1, b->_y1);
  const double y1 = std::max(b->_x1, b->_y1);

  const double x2 = std::min(b->_x2, b->_y2);
  const double y2 = std::max(b->_x2, b->_y2);

  const Exact_Point_3 p0( y0,  x1,  x2);
  const Exact_Point_3 p1( x0,  x1,  y2);
  const Exact_Point_3 p2( x0,  x1,  x2);
  const Exact_Point_3 p3( x0,  y1,  x2);
  const Exact_Point_3 p4( y0,  x1,  y2);
  const Exact_Point_3 p5( x0,  y1,  y2);
  const Exact_Point_3 p6( y0,  y1,  x2);
  const Exact_Point_3 p7( y0,  y1,  y2);
  
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

  return Nef_polyhedron_3(P);;
}

//-----------------------------------------------------------------------------
static Nef_polyhedron_3 make_tetrahedron(const csg::Tetrahedron* b)
{
  Exact_Polyhedron_3 P;
  P.make_tetrahedron( Exact_Point_3(b->x0.x(), b->x0.y(), b->x0.z()), 
		      Exact_Point_3(b->x1.x(), b->x1.y(), b->x1.z()),
		      Exact_Point_3(b->x2.x(), b->x2.y(), b->x2.z()),
		      Exact_Point_3(b->x3.x(), b->x3.y(), b->x3.z()));
  return Nef_polyhedron_3(P);
}
//-----------------------------------------------------------------------------
// // Return some vector orthogonal to a
static Point generate_orthogonal(const Point& a)
{
  const Point b(0, 1, 0);
  const Point c(0, 0, 1);
  
  // Find a vector not parallel to a.
  const Point d = (fabs(a.dot(b)) < fabs(a.dot(c))) ? b : c;
  return a.cross(d);
}
//-----------------------------------------------------------------------------
class Build_cone : public CGAL::Modifier_base<Exact_HalfedgeDS> 
{
 public:
  Build_cone(const csg::Cone* cone) : cone(cone){}

  void operator()( Exact_HalfedgeDS& hds )
  {
    const dolfin::Point axis = (cone->top - cone->bottom)/(cone->top-cone->bottom).norm();
    dolfin::Point initial = generate_orthogonal(axis);

    CGAL::Polyhedron_incremental_builder_3<Exact_HalfedgeDS> builder( hds, true);

    const int num_sides = cone->slices;
    const bool top_degenerate = near(cone->top_radius, 0.0);
    const bool bottom_degenerate = near(cone->bottom_radius, 0.0);

    const int num_vertices = (top_degenerate || bottom_degenerate) ? num_sides+2 : num_sides*2+2;

    builder.begin_surface(num_vertices, num_sides*4);

    const double delta_theta = 2.0 * DOLFIN_PI / num_sides;
    for (int i = 0; i < num_sides; ++i) 
    {
      const double theta = i*delta_theta;
      const Point rotated = initial.rotate(axis, theta);

      if (!bottom_degenerate)
      {
	const Point p = cone->bottom + rotated*cone->bottom_radius;
	const Exact_Point_3 p_(p.x(), p.y(), p.z());
	add_vertex(builder, p_);
      }

      if (!top_degenerate) 
      {
	const Point p = cone->top + rotated*cone->top_radius;
	const Exact_Point_3 p_(p.x(), p.y(), p.z());
        add_vertex(builder, p_);
      }
    }

    // The top and bottom vertices
    add_vertex(builder, Exact_Point_3(cone->bottom.x(), cone->bottom.y(), cone->bottom.z()));
    add_vertex(builder, Exact_Point_3(cone->top.x(), cone->top.y(), cone->top.z()));

    // bottom vertex has index num_vertices-2, top vertex has index num_vertices-1

    // Construct the facets on the side. 
    // Vertices must be sorted counter clockwise seen from inside.
    for (int i = 0; i < num_sides; ++i) 
    {
      if (top_degenerate) 
      {
    	std::vector<int> f;
    	f.push_back((i+1)%num_sides);
    	f.push_back(i);
    	f.push_back(num_vertices-1);
    	add_facet(builder, f);
      } else if (bottom_degenerate) 
      {
    	std::vector<int> f;
    	f.push_back( (i) );
	f.push_back( (i+1) % num_sides);
    	f.push_back(num_vertices-1);
    	add_facet(builder, f);
      } else 
      {
	//Draw the sides as triangles. 
    	const int vertex_offset = i*2;

    	// First triangle
    	std::vector<int> f;
    	f.push_back(vertex_offset);
    	f.push_back(vertex_offset+1);
    	f.push_back((vertex_offset + 2) % (num_sides*2));
    	add_facet(builder, f);

    	// Second triangle
    	std::vector<int> g;
    	g.push_back((vertex_offset + 3) % (num_sides*2));
	g.push_back((vertex_offset + 2) % (num_sides*2));
    	g.push_back(vertex_offset+1);
    	add_facet(builder, g);
      }
    }

    // Construct the bottom facet.
    if (!bottom_degenerate) 
    {
      for (int i = num_sides-1; i >= 0; i -= 1) 
      {
	std::vector<int> f;
	if (!top_degenerate)
	{
	  f.push_back(num_vertices-2);
	  f.push_back( i*2);
	  f.push_back( ( (i+1)*2) % (num_sides*2));
	} else
	{
	  f.push_back(num_vertices-2);
	  f.push_back(i);
	  f.push_back( (i+1)%num_sides );
	}
	add_facet(builder, f);
      }
    }

    // Construct the the top facet
    if (!top_degenerate) 
    {
      for (int i = 0; i < num_sides; i++)
      {
	if (!bottom_degenerate)
	{
	  std::vector<int> f;      
	  f.push_back(num_vertices-1);
	  f.push_back( ( (i+1)*2)%(num_sides*2) +1 );
	  f.push_back( i*2 + 1 );
	  add_facet(builder, f);
	} else
	{
	  std::vector<int> f;
	  f.push_back(num_vertices-2);
	  f.push_back( (i+1)%num_sides);
	  f.push_back(i);

	  add_facet(builder, f);
	}
      }
    }

    builder.end_surface();
  }
private:
  const csg::Cone* cone;
};
//-----------------------------------------------------------------------------
Nef_polyhedron_3 make_cone(const csg::Cone* c)
{
  Exact_Polyhedron_3 P;
  Build_cone builder(c);
  P.delegate(builder);
  dolfin_assert(P.is_closed());
  return Nef_polyhedron_3(P);
}
//-----------------------------------------------------------------------------
static Nef_polyhedron_3 convertSubTree(const CSGGeometry *geometry)
{
  switch (geometry->getType()) {

  case CSGGeometry::Union :
  {
    const CSGUnion* u = dynamic_cast<const CSGUnion*>(geometry);
    dolfin_assert(u);
    return convertSubTree(u->_g0.get()) + convertSubTree(u->_g1.get());
    break;
  }
  case CSGGeometry::Intersection :
  {
    const CSGIntersection* u = dynamic_cast<const CSGIntersection*>(geometry);
    dolfin_assert(u);
    return convertSubTree(u->_g0.get()) * convertSubTree(u->_g1.get());
    break;
  }
  case CSGGeometry::Difference :
  {
    const CSGDifference* u = dynamic_cast<const CSGDifference*>(geometry);
    dolfin_assert(u);
    return convertSubTree(u->_g0.get()) - convertSubTree(u->_g1.get());
    break;
  }
  case CSGGeometry::Cone :
  {
    const csg::Cone* c = dynamic_cast<const csg::Cone*>(geometry);
    dolfin_assert(c);
    return make_cone(c);
    break;
  }
  case CSGGeometry::Sphere :
  {
    const csg::Sphere* s = dynamic_cast<const csg::Sphere*>(geometry);
    dolfin_assert(s);
    return make_sphere(s);
    break;
  }
  case CSGGeometry::Box :
  {
    const csg::Box* b = dynamic_cast<const csg::Box*>(geometry);
    dolfin_assert(b);
    return make_box(b);
    break;
  }

  case CSGGeometry::Tetrahedron :
  {
    const csg::Tetrahedron* b = dynamic_cast<const csg::Tetrahedron*>(geometry);
    dolfin_assert(b);
    return make_tetrahedron(b);
    break;
  }

  default:
    dolfin_error("GeometryToCGALConverter.cpp",
		 "converting geometry to cgal polyhedron",
		 "Unhandled primitive type");
  }
    // Make compiler happy.
  return Nef_polyhedron_3();
}
//-----------------------------------------------------------------------------
void GeometryToCGALConverter::convert(const CSGGeometry& geometry, csg::Polyhedron_3& p)
{
  Nef_polyhedron_3 cgal_geometry = convertSubTree(&geometry);

  dolfin_assert(cgal_geometry.is_valid());
  dolfin_assert(cgal_geometry.is_simple());

  Exact_Polyhedron_3 p_exact;
  cgal_geometry.convert_to_polyhedron(p_exact);

  copy_to(p_exact, p);
}
