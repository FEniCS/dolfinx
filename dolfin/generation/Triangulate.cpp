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

#include <dolfin/log/log.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/Point.h>
#include "CGALMeshBuilder.h"
#include "Triangulate.h"
#include "cgal_triangulate.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void Triangulate::triangulate(Mesh& mesh, const std::vector<Point>& vertices,
                              unsigned int gdim)
{
  // Check that we have enough points (vertices) to create a triangulation
  dolfin_assert(vertices.size() > gdim);

  // Triangulate points and build mesh
  if (gdim == 2)
    triangulate2D(mesh, vertices);
  else if (gdim == 3)
    triangulate3D(mesh, vertices);
  else
  {
    dolfin_error("Triangulate.cpp",
                 "triangulate points using CGAL",
                 "No suitable triangulate function for geometric dim %d", gdim);
  }
}
//-----------------------------------------------------------------------------
void Triangulate::triangulate2D(Mesh& mesh, const std::vector<Point>& vertices)
{
  // Create vector of CGAL points
  std::vector<K::Point_2> cgal_points;
  std::vector<Point>::const_iterator p;
  for (p = vertices.begin(); p != vertices.end(); ++p)
    cgal_points.push_back(K::Point_2(p->x(), p->y()));

  // Create CGAL triangulation
  Triangulation2 t;
  t.insert(cgal_points.begin(), cgal_points.end());

  // Build DOLFIN mesh from triangulation
  CGALMeshBuilder::build(mesh, t);
}
//-----------------------------------------------------------------------------
void Triangulate::triangulate3D(Mesh& mesh, const std::vector<Point>& vertices)
{
  // Create vector of CGAL points
  std::vector<K::Point_3> cgal_points;
  std::vector<Point>::const_iterator p;
  for (p = vertices.begin(); p != vertices.end(); ++p)
    cgal_points.push_back(K::Point_3(p->x(), p->y(), p->z()));

  // Create CGAL triangulation
  Triangulation3 t;
  t.insert(cgal_points.begin(), cgal_points.end());

  // Build DOLFIN mesh from triangulation
  CGALMeshBuilder::build(mesh, t);
}
//-----------------------------------------------------------------------------
#endif
