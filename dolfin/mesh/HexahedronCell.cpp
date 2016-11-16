// Copyright (C) 2015 Chris Richardson
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

#include <algorithm>
#include <dolfin/log/log.h>
#include "Cell.h"
#include "MeshEditor.h"
#include "MeshEntity.h"
#include "Facet.h"
#include "HexahedronCell.h"
#include "Vertex.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
std::size_t HexahedronCell::dim() const
{
  return 3;
}
//-----------------------------------------------------------------------------
std::size_t HexahedronCell::num_entities(std::size_t dim) const
{
  switch (dim)
  {
  case 0:
    return 8; // vertices
  case 1:
    return 12; // edges
  case 2:
    return 6; // faces
  case 3:
    return 1; // cells
  default:
    dolfin_error("HexahedronCell.cpp",
                 "access number of entities of hexahedron cell",
                 "Illegal topological dimension (%d)", dim);
  }

  return 0;
}
//-----------------------------------------------------------------------------
std::size_t HexahedronCell::num_vertices(std::size_t dim) const
{
  switch (dim)
  {
  case 0:
    return 1; // vertices
  case 1:
    return 2; // edges
  case 2:
    return 4; // faces
  case 3:
    return 8; // cells
  default:
    dolfin_error("HexahedronCell.cpp",
                 "access number of vertices for subsimplex of hexahedron cell",
                 "Illegal topological dimension (%d)", dim);
  }

  return 0;
}
//-----------------------------------------------------------------------------
std::size_t HexahedronCell::orientation(const Cell& cell) const
{
  const Point up(0.0, 0.0, 1.0);
  return cell.orientation(up);
}
//-----------------------------------------------------------------------------
void HexahedronCell::create_entities(boost::multi_array<unsigned int, 2>& e,
                                        std::size_t dim, const unsigned int* v) const
{
  // We need to know how to create edges and faces
  switch (dim)
  {
  case 1:
    // Resize data structure
    e.resize(boost::extents[12][2]);

    // Create the 12 edges
    e[0][0]  = v[0]; e[0][1]  = v[1];
    e[1][0]  = v[2]; e[1][1]  = v[3];
    e[2][0]  = v[4]; e[2][1]  = v[5];
    e[3][0]  = v[6]; e[3][1]  = v[7];
    e[4][0]  = v[0]; e[4][1]  = v[2];
    e[5][0]  = v[1]; e[5][1]  = v[3];
    e[6][0]  = v[4]; e[6][1]  = v[6];
    e[7][0]  = v[5]; e[7][1]  = v[7];
    e[8][0]  = v[0]; e[8][1]  = v[4];
    e[9][0]  = v[1]; e[9][1]  = v[5];
    e[10][0] = v[2]; e[10][1] = v[6];
    e[11][0] = v[3]; e[11][1] = v[7];
    break;
  case 2:
    // Resize data structure
    e.resize(boost::extents[6][4]);

    // Create the 6 faces
    e[0][0] = v[0]; e[0][1] = v[2]; e[0][2] = v[4]; e[0][3] = v[6];
    e[1][0] = v[1]; e[1][1] = v[3]; e[1][2] = v[5]; e[1][3] = v[7];
    e[2][0] = v[0]; e[2][1] = v[1]; e[2][2] = v[4]; e[2][3] = v[5];
    e[3][0] = v[2]; e[3][1] = v[3]; e[3][2] = v[6]; e[3][3] = v[7];
    e[4][0] = v[0]; e[4][1] = v[1]; e[4][2] = v[2]; e[4][3] = v[3];
    e[5][0] = v[4]; e[5][1] = v[5]; e[5][2] = v[6]; e[5][3] = v[7];
    break;
  default:
    dolfin_error("HexahedronCell.cpp",
                 "create entities of tetrahedron cell",
                 "Don't know how to create entities of topological dimension %d", dim);
  }
}
//-----------------------------------------------------------------------------
double HexahedronCell::volume(const MeshEntity& cell) const
{
  if (cell.dim() != 2)
  {
    dolfin_error("HexahedronCell.cpp",
                 "compute volume (area) of cell",
                 "Illegal mesh entity");
  }

  // Get mesh geometry
  const MeshGeometry& geometry = cell.mesh().geometry();

  // Get the coordinates of the four vertices
  const unsigned int* vertices = cell.entities(0);
  const Point p0 = geometry.point(vertices[0]);
  const Point p1 = geometry.point(vertices[1]);
  const Point p2 = geometry.point(vertices[2]);
  const Point p3 = geometry.point(vertices[3]);
  const Point p4 = geometry.point(vertices[4]);
  const Point p5 = geometry.point(vertices[5]);

  dolfin_error("HexahedronCell.cpp",
               "compute volume of hexahedron",
               "Not implemented");

  return 0.0;
}
//-----------------------------------------------------------------------------
double HexahedronCell::circumradius(const MeshEntity& cell) const
{
  // Check that we get a cell
  if (cell.dim() != 2)
  {
    dolfin_error("HexahedronCell.cpp",
                 "compute circumradius of hexahedron cell",
                 "Illegal mesh entity");
  }

  dolfin_error("HexahedronCell.cpp",
               "compute circumradius of hexahedron cell",
               "Don't know how to compute diameter");

  dolfin_not_implemented();
  return 0.0;
}
//-----------------------------------------------------------------------------
double HexahedronCell::squared_distance(const Cell& cell,
                                           const Point& point) const
{
  dolfin_not_implemented();
  return 0.0;
}
//-----------------------------------------------------------------------------
double HexahedronCell::normal(const Cell& cell, std::size_t facet, std::size_t i) const
{
  return normal(cell, facet)[i];
}
//-----------------------------------------------------------------------------
Point HexahedronCell::normal(const Cell& cell, std::size_t facet) const
{
  dolfin_not_implemented();
  Point p;
  return p;
}
//-----------------------------------------------------------------------------
Point HexahedronCell::cell_normal(const Cell& cell) const
{
  dolfin_not_implemented();
  Point p;
  return p;
}
//-----------------------------------------------------------------------------
double HexahedronCell::facet_area(const Cell& cell, std::size_t facet) const
{
  // Create facet from the mesh and local facet number
  const Facet f(cell.mesh(), cell.entities(1)[facet]);

  // Get global index of vertices on the facet
  const std::size_t v0 = f.entities(0)[0];
  const std::size_t v1 = f.entities(0)[1];
  const std::size_t v2 = f.entities(0)[2];
  const std::size_t v3 = f.entities(0)[3];

  // Get mesh geometry
  const MeshGeometry& geometry = cell.mesh().geometry();

  // Need to check points are co-planar

  const Point p0 = geometry.point(v0);
  const Point p1 = geometry.point(v1);
  const Point p2 = geometry.point(v2);
  const Point p3 = geometry.point(v3);

  dolfin_not_implemented();

  return 0.0;
}
//-----------------------------------------------------------------------------
void HexahedronCell::order(Cell& cell,
                 const std::vector<std::int64_t>& local_to_global_vertex_indices) const
{
  // Not implemented
  dolfin_not_implemented();
}
//-----------------------------------------------------------------------------
bool HexahedronCell::collides(const Cell& cell, const Point& point) const
{
  dolfin_not_implemented();
  return false;
}
//-----------------------------------------------------------------------------
bool HexahedronCell::collides(const Cell& cell, const MeshEntity& entity) const
{
  dolfin_not_implemented();
  return false;
}
//-----------------------------------------------------------------------------
std::vector<double>
HexahedronCell::triangulate_intersection(const Cell& c0, const Cell& c1) const
{
  dolfin_not_implemented();
  return std::vector<double>();
}
//-----------------------------------------------------------------------------
std::string HexahedronCell::description(bool plural) const
{
  if (plural)
    return "hexahedra";
  return "hexahedron";
}
//-----------------------------------------------------------------------------
