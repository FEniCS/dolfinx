// Copyright (C) 2006-2011 Anders Logg
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
// Modified by Kristoffer Selim 2008
// Modified by Andre Massing 2010
// Modified by Jan Blechta 2013
//
// First added:  2006-06-05
// Last changed: 2016-06-10

#include <algorithm>
#include <array>

#include <dolfin/geometry/Point.h>
#include <dolfin/log/log.h>
#include "Cell.h"
#include "CellType.h"
#include "HexahedronCell.h"
#include "IntervalCell.h"
#include "MeshFunction.h"
#include "PointCell.h"
#include "QuadrilateralCell.h"
#include "TetrahedronCell.h"
#include "TriangleCell.h"
#include "Vertex.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
CellType::CellType(Type cell_type, Type facet_type)
  : _cell_type(cell_type), _facet_type(facet_type)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
CellType::~CellType()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
CellType* CellType::create(Type type)
{
  switch ( type )
  {
  case point:
    return new PointCell();
  case interval:
    return new IntervalCell();
  case triangle:
    return new TriangleCell();
  case tetrahedron:
    return new TetrahedronCell();
  case quadrilateral:
    return new QuadrilateralCell();
  case hexahedron:
    return new HexahedronCell();
  default:
    dolfin_error("CellType.cpp",
                 "create cell type",
                 "Unknown cell type (%d)", type);
  }

  return 0;
}
//-----------------------------------------------------------------------------
CellType* CellType::create(std::string type)
{
  return create(string2type(type));
}
//-----------------------------------------------------------------------------
CellType::Type CellType::string2type(std::string type)
{
  if (type == "interval")
    return interval;
  else if (type == "triangle")
    return triangle;
  else if (type == "tetrahedron")
    return tetrahedron;
  else if (type == "quadrilateral")
    return quadrilateral;
  else if (type == "hexahedron")
    return hexahedron;
  else
  {
    dolfin_error("CellType.cpp",
                 "convert string to cell type",
                 "Unknown cell type (\"%s\")", type.c_str());
  }

  return interval;
}
//-----------------------------------------------------------------------------
std::string CellType::type2string(Type type)
{
  switch (type)
  {
  case point:
    return "point";
  case interval:
    return "interval";
  case triangle:
   return "triangle";
  case tetrahedron:
    return "tetrahedron";
  case quadrilateral:
    return "quadrilateral";
  case hexahedron:
    return "hexahedron";
  default:
    dolfin_error("CellType.cpp",
                 "convert cell type to string",
                 "Unknown cell type (\"%d\")", type);
  }

  return "";
}
//-----------------------------------------------------------------------------
CellType::Type CellType::entity_type(std::size_t i) const
{
  if (i == dim())
    return _cell_type;
  else if (i == dim() - 1)
    return _facet_type;
  else if (i == 1)
    return Type::interval;
 return Type::point;
}
//-----------------------------------------------------------------------------
std::size_t CellType::orientation(const Cell& cell, const Point& up) const
{
  Point n = cell.cell_normal();
  return (n.dot(up) < 0.0 ? 1 : 0);
}
//-----------------------------------------------------------------------------
double CellType::h(const MeshEntity& entity) const
{
  // Get mesh geometry
  const MeshGeometry& geometry = entity.mesh().geometry();

  // Get number of cell vertices
  const int num_vertices = entity.num_entities(0);

  // Get the coordinates (Points) of the vertices
  const unsigned int* vertices = entity.entities(0);
  dolfin_assert(vertices);
  std::array<Point, 4> points;
  dolfin_assert(num_vertices <= 4);
  for (int i = 0; i < num_vertices; ++i)
    points[i] = geometry.point(vertices[i]);

  // Get maximum edge length
  double h = 0.0;
  for (int i = 0; i < num_vertices; ++i)
  {
    for (int j = i + 1; j < num_vertices; ++j)
      h = std::max(h, points[i].distance(points[j]));
  }

  return h;
}
//-----------------------------------------------------------------------------
double CellType::diameter(const MeshEntity& entity) const
{
  deprecation("CellType::diameter()", "2016.1",
              "Use CellType::circumradius() or CellType::h() instead.");

  return 2.0*circumradius(entity);
}
//-----------------------------------------------------------------------------
double CellType::inradius(const Cell& cell) const
{
  // Check cell type
  if (_cell_type != interval && _cell_type != triangle
      && _cell_type != tetrahedron)
  {
    dolfin_error("Cell.h",
                 "compute cell inradius",
                 "formula not implemented for non-simplicial cells");
  }

  // Pick dim
  const size_t d = dim();

  // Compute volume
  const double V = volume(cell);

  // Handle degenerate case
  if (V == 0.0)
    return 0.0;

  // Compute total area of facets
  double A = 0;
  for (std::size_t i = 0; i <= d; i++)
    A += facet_area(cell, i);

  // See Jonathan Richard Shewchuk: What Is a Good Linear Finite
  // Element?, online:
  // http://www.cs.berkeley.edu/~jrs/papers/elemj.pdf
  return d*V/A;
}
//-----------------------------------------------------------------------------
double CellType::radius_ratio(const Cell& cell) const
{
  const double r = inradius(cell);

  // Handle degenerate case
  if (r == 0.0)
    return 0.0;
  else
    return dim()*r/circumradius(cell);
}
//-----------------------------------------------------------------------------
bool CellType::ordered(const Cell& cell, const std::vector<std::size_t>&
                       local_to_global_vertex_indices) const
{
  // Get mesh topology
  const MeshTopology& topology = cell.mesh().topology();
  const std::size_t dim = topology.dim();
  const std::size_t c = cell.index();

  // Get vertices
  const std::size_t num_vertices = topology(dim, 0).size(c);
  const unsigned int* vertices = topology(dim, 0)(c);
  dolfin_assert(vertices);

  // Check that vertices are in ascending order
  if (!increasing(num_vertices, vertices, local_to_global_vertex_indices))
    return false;

  // Note the comparison below: d + 1 < dim, not d < dim - 1
  // Otherwise, d < dim - 1 will evaluate to true for dim = 0 with std::size_t

  // Check numbering of entities of positive dimension and codimension
  for (std::size_t d = 1; d + 1 < dim; d++)
  {
    // Check if entities exist, otherwise skip
    const MeshConnectivity& connectivity = topology(d, 0);
    if (connectivity.empty())
      continue;

    // Get entities
    const std::size_t num_entities = topology(dim, d).size(c);
    const unsigned int* entities = topology(dim, d)(c);

    // Iterate over entities
    for (std::size_t e = 1; e < num_entities; e++)
    {
      // Get vertices for first entity
      const std::size_t  e0 = entities[e - 1];
      const std::size_t  n0 = connectivity.size(e0);
      const unsigned int* v0 = connectivity(e0);

      // Get vertices for second entity
      const std::size_t  e1 = entities[e];
      const std::size_t  n1 = connectivity.size(e1);
      const unsigned int* v1 = connectivity(e1);

      // Check ordering of entities
      if (!increasing(n0, v0, n1, v1, num_vertices, vertices,
                      local_to_global_vertex_indices))
        return false;
    }
  }

  return true;
}
//-----------------------------------------------------------------------------
void CellType::sort_entities(std::size_t num_vertices,
                             unsigned int* local_vertices,
                             const std::vector<std::size_t>&
                             local_to_global_vertex_indices)
{
  // Two cases here, either sort vertices directly (when running in
  // serial) or sort based on the global indices (when running in
  // parallel)

  /// Comparison operator for sorting based on global indices
  class GlobalSort
  {
  public:

    GlobalSort(const std::vector<std::size_t>& local_to_global_vertex_indices)
        : g(local_to_global_vertex_indices) {}

    bool operator() (const std::size_t& l, const std::size_t& r)
    { return g[l] < g[r]; }

    const std::vector<std::size_t>& g;

  };

  // Sort on global vertex indices
  GlobalSort global_sort(local_to_global_vertex_indices);
  std::sort(local_vertices, local_vertices + num_vertices, global_sort);
}
//-----------------------------------------------------------------------------
bool CellType::increasing(std::size_t num_vertices,
                          const unsigned int* local_vertices,
                          const std::vector<std::size_t>&
                          local_to_global_vertex_indices)
{
  // Two cases here, either check vertices directly (when running in serial)
  // or check based on the global indices (when running in parallel)

  for (std::size_t v = 1; v < num_vertices; v++)
    if (local_to_global_vertex_indices[local_vertices[v - 1]] >= local_to_global_vertex_indices[local_vertices[v]])
      return false;
  return true;
}
//-----------------------------------------------------------------------------
bool CellType::increasing(std::size_t n0, const unsigned int* v0,
                          std::size_t n1, const unsigned int* v1,
                          std::size_t num_vertices,
                          const unsigned int* local_vertices,
               const std::vector<std::size_t>& local_to_global_vertex_indices)
{
  dolfin_assert(n0 == n1);
  dolfin_assert(num_vertices > n0);
  const std::size_t num_non_incident = num_vertices - n0;

  // Compute non-incident vertices for first entity
  std::vector<std::size_t> w0(num_non_incident);
  std::size_t k = 0;
  for (std::size_t i = 0; i < num_vertices; i++)
  {
    const std::size_t v = local_vertices[i];
    bool incident = false;
    for (std::size_t j = 0; j < n0; j++)
    {
      if (v0[j] == v)
      {
        incident = true;
        break;
      }
    }
    if (!incident)
      w0[k++] = v;
  }
  dolfin_assert(k == num_non_incident);

  // Compute non-incident vertices for second entity
  std::vector<std::size_t> w1(num_non_incident);
  k = 0;
  for (std::size_t i = 0; i < num_vertices; i++)
  {
    const std::size_t v = local_vertices[i];
    bool incident = false;
    for (std::size_t j = 0; j < n1; j++)
    {
      if (v1[j] == v)
      {
        incident = true;
        break;
      }
    }

    if (!incident)
      w1[k++] = v;
  }
  dolfin_assert(k == num_non_incident);

  // Compare lexicographic ordering of w0 and w1
  for (std::size_t i = 0; i < num_non_incident; i++)
  {
    if (local_to_global_vertex_indices[w0[i]]
        < local_to_global_vertex_indices[w1[i]])
    {
      return true;
    }
    else if (local_to_global_vertex_indices[w0[i]]
             > local_to_global_vertex_indices[w1[i]])
    {
      return false;
    }
  }

  return true;
}
//-----------------------------------------------------------------------------
