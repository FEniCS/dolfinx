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
// Last changed: 2012-05-30

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
typedef Explorer::Halfedge_const_handle Halfedge_const_handle;

typedef CGAL::Triangulation_vertex_base_2<Inexact_Kernel>  Vertex_base;
typedef CGAL::Constrained_triangulation_face_base_2<Inexact_Kernel> Face_base;

template <class Gt,
          class Fb >
class Enriched_face_base_2 : public Fb {
public:
  typedef Gt Geom_traits;
  typedef typename Fb::Vertex_handle Vertex_handle;
  typedef typename Fb::Face_handle Face_handle;

  template < typename TDS2 >
  struct Rebind_TDS {
    typedef typename Fb::template Rebind_TDS<TDS2>::Other Fb2;
    typedef Enriched_face_base_2<Gt,Fb2> Other;
  };

protected:
  int status;

public:
  Enriched_face_base_2(): Fb(), status(-1) {};

  Enriched_face_base_2(Vertex_handle v0,
                       Vertex_handle v1,
                       Vertex_handle v2)
    : Fb(v0,v1,v2), status(-1) {};

  Enriched_face_base_2(Vertex_handle v0,
                       Vertex_handle v1,
                       Vertex_handle v2,
                       Face_handle n0,
                       Face_handle n1,
                       Face_handle n2)
    : Fb(v0,v1,v2,n0,n1,n2), status(-1) {};

  inline
  bool is_in_domain() const { return (status%2 == 1); };

  inline
  void set_in_domain(const bool b) { status = (b ? 1 : 0); };

  inline
  void set_counter(int i) { status = i; };

  inline
  int counter() const { return status; };

  inline
  int& counter() { return status; };
}; // end class Enriched_face_base_2

typedef CGAL::Triangulation_vertex_base_2<Inexact_Kernel> Vb;
typedef CGAL::Triangulation_vertex_base_with_info_2<unsigned int, Inexact_Kernel, Vb> Vbb;
typedef Enriched_face_base_2<Inexact_Kernel, Face_base> Fb;
typedef CGAL::Triangulation_data_structure_2<Vbb, Fb> TDS;
typedef CGAL::Exact_predicates_tag Itag;
typedef CGAL::Constrained_Delaunay_triangulation_2<Inexact_Kernel, TDS, Itag> CDT;
typedef CGAL::Delaunay_mesh_size_criteria_2<CDT> Mesh_criteria_2;
typedef CGAL::Delaunay_mesher_2<CDT, Mesh_criteria_2> CGAL_Mesher_2;

typedef CDT::Vertex_handle Vertex_handle;
typedef CDT::Face_handle Face_handle;
typedef CDT::All_faces_iterator All_faces_iterator;

typedef CGAL::Polygon_2<Inexact_Kernel> Polygon_2;
typedef Inexact_Kernel::Point_2 Point_2;

using namespace dolfin;

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
Nef_polyhedron_2 make_ellipse(const csg::Ellipse* e)
{
  std::vector<Nef_point_2> points;

  for (dolfin::uint i=0; i < e->fragments; i++)
  {
    double phi = (2*DOLFIN_PI*i) / e->fragments;
    double x = e->_x0 + e->_a*cos(phi);
    double y = e->_x1 + e->_b*sin(phi);
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
  case CSGGeometry::Ellipse:
  {
    const csg::Ellipse* c = dynamic_cast<const csg::Ellipse*>(geometry);
    dolfin_assert(c);
    return make_ellipse(c);
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
  }
  return Nef_polyhedron_2();
}
//-----------------------------------------------------------------------------
// Taken from examples/Triangulation_2/polygon_triangulation.cpp in the
// CGAL source tree.
//
// Explore set of facets connected with non constrained edges,
// and attribute to each such set a counter.
// We start from facets incident to the infinite vertex, with a counter
// set to 0. Then we recursively consider the non-explored facets incident
// to constrained edges bounding the former set and increase thecounter by 1.
// Facets in the domain are those with an odd nesting level.
void mark_domains(CDT& ct,
                  CDT::Face_handle start,
                  int index,
                  std::list<CDT::Edge>& border)
{
  if (start->counter() != -1)
    return;

  std::list<CDT::Face_handle> queue;
  queue.push_back(start);

  while (!queue.empty())
  {
    CDT::Face_handle fh = queue.front();
    queue.pop_front();
    if (fh->counter() == -1)
    {
      fh->counter() = index;
      fh->set_in_domain(index%2 == 1);
      for (int i = 0; i < 3; i++)
      {
        CDT::Edge e(fh, i);
        CDT::Face_handle n = fh->neighbor(i);
        if (n->counter() == -1)
        {
          if (ct.is_constrained(e))
            border.push_back(e);
          else
            queue.push_back(n);
        }
      }
    }
  }
}
//-----------------------------------------------------------------------------
void mark_domains(CDT& cdt)
{
  for (CDT::All_faces_iterator it = cdt.all_faces_begin(); it != cdt.all_faces_end(); ++it)
    it->set_counter(-1);

  int index = 0;
  std::list<CDT::Edge> border;
  mark_domains(cdt, cdt.infinite_face(), index++, border);
  while (!border.empty())
  {
    CDT::Edge e = border.front();
    border.pop_front();
    CDT::Face_handle n = e.first->neighbor(e.second);
    if (n->counter() == -1)
      mark_domains(cdt, n, e.first->counter()+1, border);
  }
}
//-----------------------------------------------------------------------------
void CSGCGALMeshGenerator2D::generate(Mesh& mesh)
{
  Nef_polyhedron_2 cgal_geometry = convertSubTree(&geometry);

  // Create empty CGAL triangulation
  CDT cdt;

  // Explore the Nef polyhedron and insert constraints in the triangulation
  Explorer explorer = cgal_geometry.explorer();
  Face_const_iterator fit = explorer.faces_begin();
  for (; fit != explorer.faces_end(); fit++)
  {
    // Skip face if it is not part of polygon
    if (!explorer.mark(fit))
      continue;

    Halfedge_around_face_const_circulator hafc = explorer.face_cycle(fit), done(hafc);
    do {
      Vertex_handle va = cdt.insert(Point_2(to_double(hafc->vertex()->point().x()),
                                            to_double(hafc->vertex()->point().y())));
      Vertex_handle vb = cdt.insert(Point_2(to_double(hafc->next()->vertex()->point().x()),
                                            to_double(hafc->next()->vertex()->point().y())));
      cdt.insert_constraint(va, vb);
      hafc++;
    } while (hafc != done);

    Hole_const_iterator hit = explorer.holes_begin(fit);
    for (; hit != explorer.holes_end(fit); hit++)
    {
      Halfedge_around_face_const_circulator hafc(hit), done(hit);
      do {
        Vertex_handle va = cdt.insert(Point_2(to_double(hafc->vertex()->point().x()),
                                              to_double(hafc->vertex()->point().y())));
        Vertex_handle vb = cdt.insert(Point_2(to_double(hafc->next()->vertex()->point().x()),
                                              to_double(hafc->next()->vertex()->point().y())));
        cdt.insert_constraint(va, vb);
        hafc++;
      } while (hafc != done);
    }
  }

  // Mark parts that are inside and outside the domain
  mark_domains(cdt);

  // Create mesher
  CGAL_Mesher_2 mesher(cdt);

  // Add seeds for all faces in the domain
  std::list<Point_2> list_of_seeds;
  for(CDT::Finite_faces_iterator fit = cdt.finite_faces_begin();
      fit != cdt.finite_faces_end(); ++fit)
  {
    if (fit->is_in_domain())
    {
      // Calculate center of triangle and add to list of seeds
      Point_2 p0 = fit->vertex(0)->point();
      Point_2 p1 = fit->vertex(1)->point();
      Point_2 p2 = fit->vertex(2)->point();
      const double x = (p0[0] + p1[0] + p2[0]) / 3;
      const double y = (p0[1] + p1[1] + p2[1]) / 3;
      list_of_seeds.push_back(Point_2(x, y));
    }
  }
  mesher.set_seeds(list_of_seeds.begin(), list_of_seeds.end(), true);

  // Set shape and size criteria
  Mesh_criteria_2 criteria(parameters["triangle_shape_bound"],
                           parameters["cell_size"]);
  mesher.set_criteria(criteria);

  // Refine CGAL mesh/triangulation
  mesher.refine_mesh();

  // Make sure triangulation is valid
  dolfin_assert(cdt.is_valid());

  // Clear mesh
  mesh.clear();

  // Get various dimensions
  const uint gdim = cdt.finite_vertices_begin()->point().dimension();
  const uint tdim = cdt.dimension();
  const uint num_vertices = cdt.number_of_vertices();
  const uint num_cells = cdt.number_of_faces();

  // Create a MeshEditor and open
  dolfin::MeshEditor mesh_editor;
  mesh_editor.open(mesh, tdim, gdim);
  mesh_editor.init_vertices(num_vertices);
  mesh_editor.init_cells(num_cells);

  // Add vertices to mesh
  unsigned int vertex_index = 0;
  CDT::Finite_vertices_iterator cgal_vertex;
  for (cgal_vertex = cdt.finite_vertices_begin();
       cgal_vertex != cdt.finite_vertices_end(); ++cgal_vertex)
  {
    // Get vertex coordinates and add vertex to the mesh
    Point p;
    p[0] = cgal_vertex->point()[0];
    p[1] = cgal_vertex->point()[1];
    if (gdim == 3)
      p[2] = cgal_vertex->point()[2];

    // Add mesh vertex
    mesh_editor.add_vertex(vertex_index, p);

    // Attach index to vertex and increment
    cgal_vertex->info() = vertex_index++;
  }

  // Add cells to mesh
  unsigned int cell_index = 0;
  CDT::Finite_faces_iterator cgal_cell;
  for (cgal_cell = cdt.finite_faces_begin(); cgal_cell != cdt.finite_faces_end(); ++cgal_cell)
  {
    // Add cell if it is in the domain
    if (cgal_cell->is_in_domain())
    {
      mesh_editor.add_cell(cell_index++,
                           cgal_cell->vertex(0)->info(),
                           cgal_cell->vertex(1)->info(),
                           cgal_cell->vertex(2)->info());
    }
  }

  // Close mesh editor
  mesh_editor.close();

  // Build DOLFIN mesh from CGAL triangulation
  // FIXME: Why does this not work correctly?
  //CGALMeshBuilder::build(mesh, cdt);
}
//-----------------------------------------------------------------------------
