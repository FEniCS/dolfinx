// Copyright (C) 2012 Garth N. Wells
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
// First added:  2012-02-02
// Last changed:

#ifdef HAS_CGAL

#include <vector>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>
#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include <CGAL/Delaunay_mesher_2.h>
#include <CGAL/Delaunay_mesh_face_base_2.h>
#include <CGAL/Delaunay_mesh_size_criteria_2.h>
#include <CGAL/Polygon_2.h>

#include <dolfin/log/log.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshEditor.h>
#include <dolfin/mesh/Point.h>
#include "CGALMeshBuilder.h"
#include "PolygonalMeshGenerator.h"

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;

typedef CGAL::Triangulation_vertex_base_2<K> Vbase;
typedef CGAL::Triangulation_vertex_base_with_info_2<unsigned int, K, Vbase> Vb;
typedef CGAL::Delaunay_mesh_face_base_2<K> Fb;
typedef CGAL::Triangulation_data_structure_2<Vb, Fb> Tds;
typedef CGAL::Constrained_Delaunay_triangulation_2<K, Tds> CDT;
typedef CGAL::Delaunay_mesh_size_criteria_2<CDT> Criteria;
typedef CGAL::Delaunay_mesher_2<CDT, Criteria> CGAL_Mesher;

typedef CDT::Vertex_handle Vertex_handle;
typedef CDT::Point CGAL_Point;

typedef CGAL::Polygon_2<K> Polygon_2;

using namespace dolfin;

//-----------------------------------------------------------------------------
void PolygonalMeshGenerator::generate(Mesh& mesh,
                                    const std::vector<Point>& polygon_vertices,
                                    double cell_size)
{
  std::vector<CGAL_Point> cgal_points;

  // Build list of CGAL points
  std::vector<Point>::const_iterator p;
  for (p = polygon_vertices.begin(); p != polygon_vertices.end() - 1; ++p)
    cgal_points.push_back(CGAL_Point(p->x(), p->y()));

  // Create polygon
  Polygon_2 polygon(cgal_points.begin(), cgal_points.end());

  // Generate mesh
  generate(mesh, polygon, cell_size);
}
//-----------------------------------------------------------------------------
template <typename T>
void PolygonalMeshGenerator::generate(Mesh& mesh, const T& polygon,
                                      double cell_size)
{
  // Create empty CGAL triangulation
  CDT cdt;

  // Add polygon edges as triangulation constraints
  typename Polygon_2::Edge_const_iterator edge;
  for (edge = polygon.edges_begin(); edge != polygon.edges_end(); ++edge)
    cdt.insert_constraint(edge->point(0), edge->point(1));

  // Create mesher
  CGAL_Mesher mesher(cdt);

  // Refine CGAL mesh/triangulation
  mesher.set_criteria(Criteria(0.125, cell_size));
  mesher.refine_mesh();

  // Build DOLFIN mesh from CGAL triangulation
  CGALMeshBuilder::build(mesh, cdt);
}
//-----------------------------------------------------------------------------
template <typename T>
bool PolygonalMeshGenerator::is_convex(const std::vector<T>& vertices)
{
  // Create polygon and test for convexity
  Polygon_2 polygon(vertices.begin(), vertices.end());
  return polygon.is_convex();
}
//-----------------------------------------------------------------------------
#endif
