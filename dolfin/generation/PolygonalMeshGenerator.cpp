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

#include "cgal_triangulate.h"
#include <dolfin/log/log.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshEditor.h>
#include <dolfin/mesh/Point.h>
#include "CGALMeshBuilder.h"
#include "PolygonalMeshGenerator.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void PolygonalMeshGenerator::generate(Mesh& mesh,
                                    const std::vector<Point>& polygon_vertices,
                                    double cell_size)
{
  // Create empty CGAL triangulation
  CDT cdt;

  // Add polygon vertices to CGAL triangulation
  std::vector<Point>::const_iterator p;
  for (p = polygon_vertices.begin(); p != polygon_vertices.end() - 1; ++p)
  {
    CDT::Vertex_handle v0 = cdt.insert(CDT::Point(p->x(), p->y()));
    CDT::Vertex_handle v1 = cdt.insert(CDT::Point((p + 1)->x(), (p + 1)->y()));
    cdt.insert_constraint(v0, v1);
  }

  // Create mesher
  CGAL_Mesher mesher(cdt);

  // Refine mesh
  mesher.set_criteria(Criteria(0.125, cell_size));
  mesher.refine_mesh();

  // Build DOLFIN mesh from CGAL triangulation
  CGALMeshBuilder::build(mesh, cdt);
}
//-----------------------------------------------------------------------------
#endif
