// Copyright (C) 2012 Johannes Ring
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
// Last changed: 2012-05-14

#include <vector>
#include <cmath>

#include <dolfin/common/constants.h>
#include <dolfin/math/basic.h>

#include "CSGCGALMeshGenerator2D.h"
#include "CSGGeometry.h"
#include "CSGOperators.h"
#include "CSGPrimitives2D.h"
#include "CGALMeshBuilder.h"

#include <CGAL/basic.h>
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>

#include <CGAL/Gmpq.h>
#include <CGAL/Lazy_exact_nt.h>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Bounded_kernel.h>
#include <CGAL/Nef_polyhedron_2.h>
#include <CGAL/Polygon_2.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>
#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include <CGAL/Triangulation_face_base_with_info_2.h>
#include <CGAL/Delaunay_mesher_2.h>
#include <CGAL/Delaunay_mesh_face_base_2.h>
#include <CGAL/Delaunay_mesh_size_criteria_2.h>

using namespace dolfin;


typedef CGAL::Exact_predicates_exact_constructions_kernel Exact_Kernel;
typedef CGAL::Exact_predicates_inexact_constructions_kernel Inexact_Kernel;

typedef CGAL::Lazy_exact_nt<CGAL::Gmpq> FT;
typedef CGAL::Simple_cartesian<FT> EKernel;
typedef CGAL::Bounded_kernel<EKernel> Extended_kernel;
typedef CGAL::Nef_polyhedron_2<Extended_kernel> Nef_polyhedron_2;
typedef Nef_polyhedron_2::Point Nef_point_2;

typedef Nef_polyhedron_2::Explorer Explorer;
typedef Explorer::Face_const_iterator Face_const_iterator;
typedef Explorer::Hole_const_iterator Hole_const_iterator;
typedef Explorer::Halfedge_around_face_const_circulator Halfedge_around_face_const_circulator;
typedef Explorer::Vertex_const_handle Vertex_const_handle;

typedef CGAL::Triangulation_vertex_base_2<Inexact_Kernel> Vb;
typedef CGAL::Triangulation_vertex_base_with_info_2<unsigned int, Inexact_Kernel, Vb> Vbb;
typedef CGAL::Delaunay_mesh_face_base_2<Inexact_Kernel> Fb;
typedef CGAL::Triangulation_data_structure_2<Vbb, Fb> TDS;
typedef CGAL::Constrained_Delaunay_triangulation_2<Inexact_Kernel, TDS> CDT;
typedef CGAL::Delaunay_mesh_size_criteria_2<CDT> Mesh_criteria_2;
typedef CGAL::Delaunay_mesher_2<CDT, Mesh_criteria_2> CGAL_Mesher_2;

typedef CGAL::Polygon_2<Inexact_Kernel> Polygon_2;
typedef Inexact_Kernel::Point_2 Point_2;


//-----------------------------------------------------------------------------
CSGCGALMeshGenerator2D::CSGCGALMeshGenerator2D(const CSGGeometry& geometry)
  : geometry(geometry)
{
  parameters = default_parameters();
}
//-----------------------------------------------------------------------------
CSGCGALMeshGenerator2D::~CSGCGALMeshGenerator2D() {}
//-----------------------------------------------------------------------------
Nef_polyhedron_2 make_circle(const csg::Circle* c)
{
  std::vector<Nef_point_2> points;

  for (dolfin::uint i=0; i < c->fragments; i++)
  {
    double phi = (2*DOLFIN_PI*i) / c->fragments;
    double x, y;
    if (c->_r > 0)
    {
      x = c->_x0 + c->_r*cos(phi);
      y = c->_x1 + c->_r*sin(phi);
    }
    else
    {
      x = 0;
      y = 0;
    }
    points.push_back(Nef_point_2(x, y));
  }

  return Nef_polyhedron_2(points.begin(), points.end(),
                          Nef_polyhedron_2::INCLUDED);
}
//-----------------------------------------------------------------------------
Nef_polyhedron_2 make_rectangle(const csg::Rectangle* r)
{
  const double x0 = std::min(r->_x0, r->_y0);
  const double y0 = std::max(r->_x0, r->_y0);

  const double x1 = std::min(r->_x1, r->_y1);
  const double y1 = std::max(r->_x1, r->_y1);

  std::vector<Nef_point_2> points;
  points.push_back(Nef_point_2(x0, x1));
  points.push_back(Nef_point_2(y0, x1));
  points.push_back(Nef_point_2(y0, y1));
  points.push_back(Nef_point_2(x0, y1));

  return Nef_polyhedron_2(points.begin(), points.end(),
                          Nef_polyhedron_2::INCLUDED);
}
//-----------------------------------------------------------------------------
Nef_polyhedron_2 make_polygon(const csg::Polygon* p)
{
  std::vector<Nef_point_2> points;
  std::vector<Point>::const_iterator v;
  for (v = p->vertices.begin(); v != p->vertices.end(); ++v)
    points.push_back(Nef_point_2(v->x(), v->y()));

  return Nef_polyhedron_2(points.begin(), points.end(),
                          Nef_polyhedron_2::INCLUDED);
}
//-----------------------------------------------------------------------------
static Nef_polyhedron_2 convertSubTree(const CSGGeometry *geometry)
{
  switch (geometry->getType()) {
  case CSGGeometry::Union:
  {
    const CSGUnion* u = dynamic_cast<const CSGUnion*>(geometry);
    dolfin_assert(u);
    return convertSubTree(u->_g0.get()) + convertSubTree(u->_g1.get());
    break;
  }
  case CSGGeometry::Intersection:
  {
    const CSGIntersection* u = dynamic_cast<const CSGIntersection*>(geometry);
    dolfin_assert(u);
    return convertSubTree(u->_g0.get()) * convertSubTree(u->_g1.get());
    break;
  }
  case CSGGeometry::Difference:
  {
    const CSGDifference* u = dynamic_cast<const CSGDifference*>(geometry);
    dolfin_assert(u);
    return convertSubTree(u->_g0.get()) - convertSubTree(u->_g1.get());
    break;
  }
  case CSGGeometry::Circle:
  {
    const csg::Circle* c = dynamic_cast<const csg::Circle*>(geometry);
    dolfin_assert(c);
    return make_circle(c);
    break;
  }
  case CSGGeometry::Rectangle:
  {
    const csg::Rectangle* r = dynamic_cast<const csg::Rectangle*>(geometry);
    dolfin_assert(r);
    return make_rectangle(r);
    break;
  }
  case CSGGeometry::Polygon:
  {
    const csg::Polygon* p = dynamic_cast<const csg::Polygon*>(geometry);
    dolfin_assert(p);
    return make_polygon(p);
    break;
  }
  default:
    dolfin_error("CSGCGALMeshGenerator2D.cpp",
                 "converting geometry to cgal polyhedron",
                 "Unhandled primitive type");
    // Make compiler happy
  }
  return Nef_polyhedron_2();
}
//-----------------------------------------------------------------------------
void insert_polygon(CDT& cdt, const Polygon_2& polygon)
{
  if (polygon.is_empty())
    return;

  CDT::Vertex_handle v_prev = cdt.insert(*CGAL::cpp0x::prev(polygon.vertices_end()));
  for (Polygon_2::Vertex_iterator vit = polygon.vertices_begin();
       vit != polygon.vertices_end(); ++vit)
  {
    CDT::Vertex_handle vh = cdt.insert(*vit);
    cdt.insert_constraint(vh,v_prev);
    v_prev = vh;
  }
}
//-----------------------------------------------------------------------------
void CSGCGALMeshGenerator2D::generate(Mesh& mesh)
{
  //Nef_polyhedron_2 cgal_geometry = geometry.get_cgal_type_2D();
  Nef_polyhedron_2 cgal_geometry = convertSubTree(&geometry);

  // Create empty CGAL triangulation
  CDT cdt;

  // Explore the Nef polyhedron and insert constraints in the triangulation
  Explorer explorer = cgal_geometry.explorer();
  Face_const_iterator fit = explorer.faces_begin();
  for (; fit != explorer.faces_end(); fit++)
  {
    // Skip face if it is not part of polygon
    if (! explorer.mark(fit))
      continue;

    Polygon_2 polygon;
    Halfedge_around_face_const_circulator hafc = explorer.face_cycle(fit), done(hafc);
    do {
      Vertex_const_handle vh = explorer.target(hafc);
      polygon.push_back(Point_2(to_double(explorer.point(vh).x()),
                                to_double(explorer.point(vh).y())));
      hafc++;
    } while(hafc != done);
    insert_polygon(cdt, polygon);

    // FIXME: Holes must be marked as not part of the mesh domain
    Hole_const_iterator hit = explorer.holes_begin(fit);
    for (; hit != explorer.holes_end(fit); hit++)
    {
      Polygon_2 hole;
      Halfedge_around_face_const_circulator hafc(hit), done(hit);
      do {
        Vertex_const_handle vh = explorer.target(hafc);
        hole.push_back(Point_2(to_double(explorer.point(vh).x()),
                               to_double(explorer.point(vh).y())));
        hafc++;
      } while(hafc != done);
      insert_polygon(cdt, hole);
    }
  }

  // Create mesher
  CGAL_Mesher_2 mesher(cdt);

  Mesh_criteria_2 criteria(parameters["triangle_shape_bound"],
                           parameters["cell_size"]);

  // Refine CGAL mesh/triangulation
  mesher.set_criteria(criteria);
  mesher.refine_mesh();

  dolfin_assert(cdt.is_valid());

  // Build DOLFIN mesh from CGAL triangulation
  CGALMeshBuilder::build(mesh, cdt);
}
//-----------------------------------------------------------------------------
