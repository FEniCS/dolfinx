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

#ifdef HAS_CGAL

#include <limits>
#include <CGAL/Polyhedron_incremental_builder_3.h>

#include "GeometryToCGALConverter.h"
#include "CSGGeometry.h"
#include "CSGOperators.h"
#include "PolyhedronUtils.h"
#include "CSGPrimitives3D.h"
#include <dolfin/geometry/Point.h>
#include <dolfin/log/LogStream.h>
#include <dolfin/math/basic.h>
#include "cgal_csg3d.h"

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
    : _in_poly(in_poly) {}

  void operator()(typename Polyhedron_output::HalfedgeDS& out_hds)
  {
    typedef typename Polyhedron_output::HalfedgeDS Output_HDS;
    typedef typename Polyhedron_input::HalfedgeDS Input_HDS;

    CGAL::Polyhedron_incremental_builder_3<Output_HDS> builder(out_hds);

    typedef typename Polyhedron_input::Vertex_const_iterator Vertex_const_iterator;
    typedef typename Polyhedron_input::Facet_const_iterator  Facet_const_iterator;
    typedef typename Polyhedron_input::Halfedge_around_facet_const_circulator HFCC;

    builder.begin_surface(_in_poly.size_of_vertices(),
                          _in_poly.size_of_facets(),
                          _in_poly.size_of_halfedges());

    for(Vertex_const_iterator
      vi = _in_poly.vertices_begin(), end = _in_poly.vertices_end();
      vi != end ; ++vi)
    {
      typename Polyhedron_output::Point_3 p(::CGAL::to_double( vi->point().x()),
	::CGAL::to_double( vi->point().y()),
	::CGAL::to_double( vi->point().z()));
      builder.add_vertex(p);
    }

    typedef CGAL::Inverse_index<Vertex_const_iterator> Index;
    Index index(_in_poly.vertices_begin(), _in_poly.vertices_end());

    for(Facet_const_iterator
      fi = _in_poly.facets_begin(), end = _in_poly.facets_end();
      fi != end; ++fi)
    {
      HFCC hc = fi->facet_begin();
      HFCC hc_end = hc;
      builder.begin_facet ();
      do
      {
        builder.add_vertex_to_facet(index[hc->vertex()]);
        ++hc;
      } while( hc != hc_end);
      builder.end_facet();
    }
    builder.end_surface();
  } // end operator()(..)
private:
  const Polyhedron_input& _in_poly;
}; // end Copy_polyhedron_to<>

template <class Poly_A, class Poly_B>
void copy_to(const Poly_A& poly_a, Poly_B& poly_b)
{
  Copy_polyhedron_to<Poly_A, Poly_B> modifier(poly_a);
  poly_b.delegate(modifier);
  CGAL_assertion(poly_b.is_valid());
}
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
      cout << "Vertex: " << (*it) << endl;

    if (builder.test_facet(vertices.begin(), vertices.end()))
      cout << "Facet ok, size: " << vertices.size() << endl;
    else
      cout << "Facet not ok" << endl;
  }

  builder.begin_facet();
  for (std::vector<int>::iterator it=vertices.begin(); it != vertices.end(); it++)
    builder.add_vertex_to_facet(*it);
  builder.end_facet();

  if (print)
    cout << "End facet" << endl;
  facet_no++;
}
//-----------------------------------------------------------------------------
static void add_vertex(CGAL::Polyhedron_incremental_builder_3<csg::Exact_HalfedgeDS>& builder,
		       const csg::Exact_Point_3& point, bool print=false)
{
  static int vertex_no = 0;
  if (print)
    std::cout << "Adding vertex " << vertex_no << " at " << point << std::endl;

  builder.add_vertex(point);
  vertex_no++;
}
//-----------------------------------------------------------------------------
// Sphere
//-----------------------------------------------------------------------------
class Build_sphere : public CGAL::Modifier_base<csg::Exact_HalfedgeDS>
{
 public:
  Build_sphere(const Sphere& sphere) : _sphere(sphere) {}

  void operator()( csg::Exact_HalfedgeDS& hds )
  {
    const std::size_t num_slices = _sphere._slices;
    const std::size_t num_sectors = _sphere._slices*2 + 1;

    const dolfin::Point top = _sphere.c + Point(_sphere.r, 0, 0);
    const dolfin::Point bottom = _sphere.c - Point(_sphere.r, 0, 0);
    const dolfin::Point axis = Point(1, 0, 0);

    const int num_vertices = num_slices*num_sectors+2;
    const int num_facets = num_sectors*2*num_slices;

    CGAL::Polyhedron_incremental_builder_3<csg::Exact_HalfedgeDS> builder( hds, true );

    builder.begin_surface(num_vertices, num_facets);

    const Point slice_rotation_axis(0, 1, 0);

    for (std::size_t i = 0; i < num_slices; i++)
    {
      const Point sliced = axis.rotate(slice_rotation_axis, (i+1)*DOLFIN_PI/(num_slices+1));
      for (std::size_t j = 0; j < num_sectors; j++)
      {
        const Point direction = sliced.rotate(axis, j*2.0*DOLFIN_PI/num_sectors);
        const Point v = _sphere.c + direction*_sphere.r;
        add_vertex(builder, csg::Exact_Point_3 (v.x(), v.y(), v.z()));
      }
    }

    // Add bottom has index num_vertices-1, top has index num_vertices-2
    add_vertex(builder, csg::Exact_Point_3(top.x(), top.y(), top.z()));
    add_vertex(builder, csg::Exact_Point_3(bottom.x(), bottom.y(), bottom.z()));

    // Add the side facets
    for (std::size_t i = 0; i < num_slices-1; i++)
    {
      for (std::size_t j = 0; j < num_sectors; j++)
      {
        const std::size_t offset1 = i*num_sectors;
        const std::size_t offset2 = (i+1)*num_sectors;

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
    const std::size_t top_offset = num_sectors*(num_slices-1);
    for (std::size_t i = 0; i < num_sectors; i++)
    {
      {
        // Bottom facet
        std::vector<int> f;
        f.push_back( num_vertices-2 );
        f.push_back( (i+1)%num_sectors );
        f.push_back(i);
        add_facet(builder, f);
      }

      {
        // Top facet
        std::vector<int> f;
        f.push_back( num_vertices-1 );
        f.push_back( top_offset + (i%num_sectors) );
        f.push_back( top_offset + (i+1)%num_sectors );
        add_facet(builder, f);
      }
    }
    builder.end_surface();
  }

  private:
  const Sphere& _sphere;
};
//-----------------------------------------------------------------------------
static void make_sphere(const Sphere* s, csg::Exact_Polyhedron_3& P)
{
  Build_sphere builder(*s);
  P.delegate(builder);
  dolfin_assert(P.is_valid());
  dolfin_assert(P.is_closed());
}
//-----------------------------------------------------------------------------
class Build_box : public CGAL::Modifier_base<csg::Exact_HalfedgeDS>
{
 public:
  Build_box(const Box* box) : _box(box) {}

  void operator()( csg::Exact_HalfedgeDS& hds )
  {
    CGAL::Polyhedron_incremental_builder_3<csg::Exact_HalfedgeDS> builder(hds, true);

    builder.begin_surface(8, 12);

    const double x0 = std::min(_box->_x0, _box->_y0);
    const double y0 = std::max(_box->_x0, _box->_y0);

    const double x1 = std::min(_box->_x1, _box->_y1);
    const double y1 = std::max(_box->_x1, _box->_y1);

    const double x2 = std::min(_box->_x2, _box->_y2);
    const double y2 = std::max(_box->_x2, _box->_y2);

    add_vertex(builder, csg::Exact_Point_3(y0, x1, x2));
    add_vertex(builder, csg::Exact_Point_3(x0, x1, y2));
    add_vertex(builder, csg::Exact_Point_3(x0, x1, x2));
    add_vertex(builder, csg::Exact_Point_3(x0, y1, x2));
    add_vertex(builder, csg::Exact_Point_3(y0, x1, y2));
    add_vertex(builder, csg::Exact_Point_3(x0, y1, y2));
    add_vertex(builder, csg::Exact_Point_3(y0, y1, x2));
    add_vertex(builder, csg::Exact_Point_3(y0, y1, y2));

    {
      std::vector<int> f;
      f.push_back(1);
      f.push_back(2);
      f.push_back(3);
      add_facet(builder, f);
    }

    {
      std::vector<int> f;
      f.push_back(1);
      f.push_back(3);
      f.push_back(5);
      add_facet(builder, f);
    }

    {
      std::vector<int> f;
      f.push_back(1);
      f.push_back(5);
      f.push_back(4);
      add_facet(builder, f);
    }

    {
      std::vector<int> f;
      f.push_back(4);
      f.push_back(5);
      f.push_back(7);
      add_facet(builder, f);
    }

    {
      std::vector<int> f;
      f.push_back(4);
      f.push_back(7);
      f.push_back(0);
      add_facet(builder, f);
    }

    {
      std::vector<int> f;
      f.push_back(0);
      f.push_back(7);
      f.push_back(6);
      add_facet(builder, f);
    }

    {
      std::vector<int> f;
      f.push_back(0);
      f.push_back(6);
      f.push_back(2);
      add_facet(builder, f);
    }

    {
      std::vector<int> f;
      f.push_back(2);
      f.push_back(6);
      f.push_back(3);
      add_facet(builder, f);
    }

    {
      std::vector<int> f;
      f.push_back(7);
      f.push_back(5);
      f.push_back(6);
      add_facet(builder, f);
    }

    {
      std::vector<int> f;
      f.push_back(6);
      f.push_back(5);
      f.push_back(3);
      add_facet(builder, f);
    }

    {
      std::vector<int> f;
      f.push_back(1);
      f.push_back(4);
      f.push_back(2);
      add_facet(builder, f);
    }

    {
      std::vector<int> f;
      f.push_back(2);
      f.push_back(4);
      f.push_back(0);
      add_facet(builder, f);
    }

    builder.end_surface();
  }

  const Box* _box;
};
//-----------------------------------------------------------------------------
static void make_box(const Box* b, csg::Exact_Polyhedron_3& P)
{
  Build_box builder(b);
  P.delegate(builder);
  dolfin_assert(P.is_closed());
  dolfin_assert(P.is_valid());
}
//-----------------------------------------------------------------------------
static void make_tetrahedron(const Tetrahedron* b, csg::Exact_Polyhedron_3& P)
{
  P.make_tetrahedron(csg::Exact_Point_3(b->_x0.x(), b->_x0.y(), b->_x0.z()),
                     csg::Exact_Point_3(b->_x1.x(), b->_x1.y(), b->_x1.z()),
                     csg::Exact_Point_3(b->_x2.x(), b->_x2.y(), b->_x2.z()),
                     csg::Exact_Point_3(b->_x3.x(), b->_x3.y(), b->_x3.z()));
}
//-----------------------------------------------------------------------------
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
  Build_cone(const Cone* cone) : _cone(cone) {}

  void operator()(csg::Exact_HalfedgeDS& hds)
  {
    const dolfin::Point axis = (_cone->_top - _cone->_bottom)/(_cone->_top - _cone->_bottom).norm();
    dolfin::Point initial = generate_orthogonal(axis);

    CGAL::Polyhedron_incremental_builder_3<csg::Exact_HalfedgeDS> builder(hds, true);

    const int num_sides = _cone->_slices;
    const bool top_degenerate = near(_cone->_top_radius, 0.0);
    const bool bottom_degenerate = near(_cone->_bottom_radius, 0.0);

    const int num_vertices = (top_degenerate || bottom_degenerate) ? num_sides+2 : num_sides*2+2;

    builder.begin_surface(num_vertices, num_sides*4);

    const double delta_theta = 2.0 * DOLFIN_PI / num_sides;
    for (int i = 0; i < num_sides; ++i)
    {
      const double theta = i*delta_theta;
      const Point rotated = initial.rotate(axis, theta);
      if (!bottom_degenerate)
      {
        const Point p = _cone->_bottom + rotated*_cone->_bottom_radius;
        const csg::Exact_Point_3 p_(p.x(), p.y(), p.z());
        add_vertex(builder, p_);
      }
      if (!top_degenerate)
      {
        const Point p = _cone->_top + rotated*_cone->_top_radius;
        const csg::Exact_Point_3 p_(p.x(), p.y(), p.z());
        add_vertex(builder, p_);
      }
    }

    // The top and bottom vertices
    add_vertex(builder, csg::Exact_Point_3(_cone->_bottom.x(), _cone->_bottom.y(),
                                           _cone->_bottom.z()));
    add_vertex(builder, csg::Exact_Point_3(_cone->_top.x(), _cone->_top.y(),
                                           _cone->_top.z()));

    // bottom vertex has index num_vertices-2, top vertex has index num_vertices-1

    // Construct the facets on the side.
    // Vertices must be sorted counter clockwise seen from inside.
    for (int i = 0; i < num_sides; ++i)
    {
      if (top_degenerate)
      {
        std::vector<int> f;
        f.push_back((i + 1)%num_sides);
        f.push_back(i);
        f.push_back(num_vertices - 1);
        add_facet(builder, f);
      }
      else if (bottom_degenerate)
      {
        std::vector<int> f;
        f.push_back( (i) );
        f.push_back( (i + 1) % num_sides);
        f.push_back(num_vertices - 1);
        add_facet(builder, f);
      }
      else
      {
        //Draw the sides as triangles.
        const int vertex_offset = i*2;

        // First triangle
        std::vector<int> f;
        f.push_back(vertex_offset);
        f.push_back(vertex_offset + 1);
        f.push_back((vertex_offset + 2) % (num_sides*2));
        add_facet(builder, f);

        // Second triangle
        std::vector<int> g;
        g.push_back((vertex_offset + 3) % (num_sides*2));
        g.push_back((vertex_offset + 2) % (num_sides*2));
        g.push_back(vertex_offset + 1);
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
        }
        else
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
        }
        else
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
  const Cone* _cone;
};
//-----------------------------------------------------------------------------
static void make_cone(const Cone* c, csg::Exact_Polyhedron_3& P)
{
  Build_cone builder(c);
  P.delegate(builder);
  dolfin_assert(P.is_closed());
  dolfin_assert(P.is_valid());
}
//-----------------------------------------------------------------------------
static void make_surface3D(const Surface3D* s, csg::Exact_Polyhedron_3& P)
{
  dolfin_assert(s);
  PolyhedronUtils::readSurfaceFile(s->_filename, P);
}
//-----------------------------------------------------------------------------
static boost::shared_ptr<csg::Nef_polyhedron_3>
convertSubTree(const CSGGeometry *geometry)
{
  switch (geometry->getType())
  {
    case CSGGeometry::Union :
    {
      const CSGUnion* u = dynamic_cast<const CSGUnion*>(geometry);
      dolfin_assert(u);
      boost::shared_ptr<csg::Nef_polyhedron_3> g0 = convertSubTree(u->_g0.get());
      boost::shared_ptr<csg::Nef_polyhedron_3> g1 = convertSubTree(u->_g1.get());
      (*g0) += (*g1);
      return g0;

      break;
    }
    case CSGGeometry::Intersection :
    {
      const CSGIntersection* u = dynamic_cast<const CSGIntersection*>(geometry);
      dolfin_assert(u);
      boost::shared_ptr<csg::Nef_polyhedron_3> g0 = convertSubTree(u->_g0.get());
      boost::shared_ptr<csg::Nef_polyhedron_3> g1 = convertSubTree(u->_g1.get());
      (*g0) *= (*g1);
      return g0;
      break;
    }
    case CSGGeometry::Difference :
    {
      const CSGDifference* u = dynamic_cast<const CSGDifference*>(geometry);
      dolfin_assert(u);
      boost::shared_ptr<csg::Nef_polyhedron_3> g0 = convertSubTree(u->_g0.get());
      boost::shared_ptr<csg::Nef_polyhedron_3> g1 = convertSubTree(u->_g1.get());
      (*g0) -= (*g1);
      return g0;
      break;
    }
    case CSGGeometry::Cone :
    {
      const Cone* c = dynamic_cast<const Cone*>(geometry);
      dolfin_assert(c);
      csg::Exact_Polyhedron_3 P;
      make_cone(c, P);
      return boost::shared_ptr<csg::Nef_polyhedron_3>(new csg::Nef_polyhedron_3(P));
      break;
    }
    case CSGGeometry::Sphere :
    {
      const Sphere* s = dynamic_cast<const Sphere*>(geometry);
      dolfin_assert(s);
      csg::Exact_Polyhedron_3 P;
      make_sphere(s, P);
      return boost::shared_ptr<csg::Nef_polyhedron_3>(new csg::Nef_polyhedron_3(P));
      break;
    }
    case CSGGeometry::Box :
    {
      const Box* b = dynamic_cast<const Box*>(geometry);
      dolfin_assert(b);
      csg::Exact_Polyhedron_3 P;
      make_box(b, P);
      return boost::shared_ptr<csg::Nef_polyhedron_3>(new csg::Nef_polyhedron_3(P));
      break;
    }

    case CSGGeometry::Tetrahedron :
    {
      const Tetrahedron* b = dynamic_cast<const Tetrahedron*>(geometry);
      dolfin_assert(b);
      csg::Exact_Polyhedron_3 P;
      make_tetrahedron(b, P);
      return boost::shared_ptr<csg::Nef_polyhedron_3>(new csg::Nef_polyhedron_3(P));
      break;
    }
    case CSGGeometry::Surface3D :
    {
      const Surface3D* b = dynamic_cast<const Surface3D*>(geometry);
      dolfin_assert(b);
      csg::Exact_Polyhedron_3 P;
      make_surface3D(b, P);
      return boost::shared_ptr<csg::Nef_polyhedron_3>(new csg::Nef_polyhedron_3(P));
      break;
    }
    default:
      dolfin_error("GeometryToCGALConverter.cpp",
		   "converting geometry to cgal polyhedron",
		   "Unhandled primitive type");
  }

  // Make compiler happy.
  return boost::shared_ptr<csg::Nef_polyhedron_3>(new csg::Nef_polyhedron_3);
}
//-----------------------------------------------------------------------------
void GeometryToCGALConverter::convert(const CSGGeometry& geometry,
                                      csg::Polyhedron_3 &p,
                                      bool remove_degenerated)
{
  csg::Exact_Polyhedron_3 P;

  // If the tree has only one node, we don't have to convert to Nef
  // polyhedrons for csg manipulations
  if (!geometry.is_operator())
  {
    switch (geometry.getType())
    {

    case CSGGeometry::Cone :
    {
      const Cone* c = dynamic_cast<const Cone*>(&geometry);
      dolfin_assert(c);
      make_cone(c, P);
      break;
    }
    case CSGGeometry::Sphere :
    {
    const Sphere* s = dynamic_cast<const Sphere*>(&geometry);
    dolfin_assert(s);
    make_sphere(s, P);
    break;
    }
    case CSGGeometry::Box :
    {
      const Box* b = dynamic_cast<const Box*>(&geometry);
      dolfin_assert(b);
      make_box(b, P);
      break;
    }

    case CSGGeometry::Tetrahedron :
    {
      const Tetrahedron* b = dynamic_cast<const Tetrahedron*>(&geometry);
      dolfin_assert(b);
      make_tetrahedron(b, P);
      break;
    }
    case CSGGeometry::Surface3D :
    {
      const Surface3D* b = dynamic_cast<const Surface3D*>(&geometry);
      dolfin_assert(b);
      make_surface3D(b, P);
      break;
    }
    default:
      dolfin_error("GeometryToCGALConverter.cpp",
                   "converting geometry to cgal polyhedron",
                   "Unhandled primitive type");
    }
  }
  else
  {
    cout << "Convert to nef polyhedron" << endl;
    boost::shared_ptr<csg::Nef_polyhedron_3> cgal_geometry
      = convertSubTree(&geometry);
    dolfin_assert(cgal_geometry->is_valid());
    dolfin_assert(cgal_geometry->is_simple());
    cgal_geometry->convert_to_polyhedron(P);
  }

  if (remove_degenerated)
  {
    cout << "Removing degenerated facets" << endl;
    PolyhedronUtils::remove_degenerate_facets(P, DOLFIN_SQRT_EPS);
  }

  copy_to(P, p);

  cout << "Number of vertices: " << p.size_of_vertices() << endl;
  cout << "Number of facets:   " << p.size_of_facets() << endl;
}
//-----------------------------------------------------------------------------
#endif
