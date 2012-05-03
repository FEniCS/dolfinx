// Copyright (C) 2012 Anders Logg, Benjamin Kehlet, Johannes Ring
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
// First added:  2012-01-01
// Last changed: 2012-05-03

#include "cgal_csg.h"

#include <dolfin/log/log.h>
#include "CSGMeshGenerator.h"
#include "CSGGeometry.h"
#include "CGALMeshBuilder.h"

using namespace dolfin;

#ifdef HAS_CGAL

typedef CGAL::Polygon_2<csg::Inexact_Kernel> Polygon_2;
typedef csg::Inexact_Kernel::Point_2 Point_2;

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
void CSGMeshGenerator::generate(Mesh& mesh,
                                const CSGGeometry& geometry)
{
  if (geometry.dim() == 2)
  {
    generate_2d(mesh, geometry);
  }
  else if (geometry.dim() == 3)
  {
    generate_3d(mesh, geometry);
  }
  else
  {
    dolfin_error("CSGMeshGenerator.cpp",
                 "create mesh from CSG geometry",
                 "Unhandled geometry dimension %d", geometry.dim());
  }
}
//-----------------------------------------------------------------------------
void insert_polygon(csg::CDT& cdt, const Polygon_2& polygon)
{
  if (polygon.is_empty())
    return;

  csg::CDT::Vertex_handle v_prev = cdt.insert(*CGAL::cpp0x::prev(polygon.vertices_end()));
  for (Polygon_2::Vertex_iterator vit = polygon.vertices_begin();
       vit != polygon.vertices_end(); ++vit)
  {
    csg::CDT::Vertex_handle vh = cdt.insert(*vit);
    cdt.insert_constraint(vh,v_prev);
    v_prev = vh;
  }
}
//-----------------------------------------------------------------------------
void CSGMeshGenerator::generate_2d(Mesh& mesh,
                                   const CSGGeometry& geometry)
{
  csg::Nef_polyhedron_2 cgal_geometry = geometry.get_cgal_type_2D();

  // Create empty CGAL triangulation
  csg::CDT cdt;

  // Explore the Nef polyhedron and insert constraints in the triangulation
  csg::Explorer explorer = cgal_geometry.explorer();
  csg::Face_const_iterator fit = explorer.faces_begin();
  for (; fit != explorer.faces_end(); fit++)
  {
    // Skip face if it is not part of polygon
    if (! explorer.mark(fit))
      continue;

    Polygon_2 polygon;
    csg::Halfedge_around_face_const_circulator hafc = explorer.face_cycle(fit), done(hafc);
    do {
      csg::Vertex_const_handle vh = explorer.target(hafc);
      polygon.push_back(Point_2(to_double(explorer.point(vh).x()),
                                to_double(explorer.point(vh).y())));
      hafc++;
    } while(hafc != done);
    insert_polygon(cdt, polygon);

    // FIXME: Holes must be marked as not part of the mesh domain
    csg::Hole_const_iterator hit = explorer.holes_begin(fit);
    for (; hit != explorer.holes_end(fit); hit++)
    {
      Polygon_2 hole;
      csg::Halfedge_around_face_const_circulator hafc(hit), done(hit);
      do {
        csg::Vertex_const_handle vh = explorer.target(hafc);
        hole.push_back(Point_2(to_double(explorer.point(vh).x()),
                               to_double(explorer.point(vh).y())));
        hafc++;
      } while(hafc != done);
      insert_polygon(cdt, hole);
    }
  }

  // Create mesher
  csg::CGAL_Mesher_2 mesher(cdt);

  csg::Mesh_criteria_2 criteria(0.125, 0.25);

  // Refine CGAL mesh/triangulation
  mesher.set_criteria(criteria);
  mesher.refine_mesh();

  dolfin_assert(cdt.is_valid());

  // Build DOLFIN mesh from CGAL triangulation
  CGALMeshBuilder::build(mesh, cdt);
}
//-----------------------------------------------------------------------------
void CSGMeshGenerator::generate_3d(Mesh& mesh,
                                const CSGGeometry& geometry)
{
  csg::Nef_polyhedron_3 cgal_geometry = geometry.get_cgal_type_3D();

  dolfin_assert(cgal_geometry.is_valid());
  dolfin_assert(cgal_geometry.is_simple());

  csg::Exact_Polyhedron_3 p;
  cgal_geometry.convert_to_polyhedron(p);

  csg::Polyhedron_3 p_inexact;
  copy_to(p, p_inexact);

  // Create domain
  csg::Mesh_domain domain(p_inexact);

  csg::Mesh_criteria criteria(CGAL::parameters::facet_angle=25,
  				CGAL::parameters::facet_size=0.15,
  				CGAL::parameters::facet_distance=0.008,
  				CGAL::parameters::cell_radius_edge_ratio=3);

  // Generate CGAL mesh
  csg::C3t3 c3t3 = CGAL::make_mesh_3<csg::C3t3>(domain, criteria);

  // Build DOLFIN mesh from CGAL mesh/triangulation
  CGALMeshBuilder::build_from_mesh(mesh, c3t3);
}
//-----------------------------------------------------------------------------
#else
void CSGMeshGenerator::generate(Mesh& mesh,
                                const CSGGeometry& geometry)
{
  dolfin_error("CSGMeshGenerator.cpp",
	       "create mesh from CSG geometry",
	       "Mesh generation not available. Dolfin has been compiled without CGAL.");
}
#endif
