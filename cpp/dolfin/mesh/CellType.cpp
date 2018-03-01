// Copyright (C) 2006-2011 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "CellType.h"
#include "Cell.h"
#include "HexahedronCell.h"
#include "IntervalCell.h"
#include "MeshFunction.h"
#include "PointCell.h"
#include "QuadrilateralCell.h"
#include "TetrahedronCell.h"
#include "TriangleCell.h"
#include "Vertex.h"
#include <algorithm>
#include <array>
#include <dolfin/geometry/Point.h>
#include <dolfin/log/log.h>

using namespace dolfin;
using namespace dolfin::mesh;

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
  switch (type)
  {
  case Type::point:
    return new PointCell();
  case Type::interval:
    return new IntervalCell();
  case Type::triangle:
    return new TriangleCell();
  case Type::tetrahedron:
    return new TetrahedronCell();
  case Type::quadrilateral:
    return new QuadrilateralCell();
  case Type::hexahedron:
    return new HexahedronCell();
  default:
    log::dolfin_error("CellType.cpp", "create cell type",
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
  if (type == "point")
    return Type::point;
  else if (type == "interval")
    return Type::interval;
  else if (type == "triangle")
    return Type::triangle;
  else if (type == "tetrahedron")
    return Type::tetrahedron;
  else if (type == "quadrilateral")
    return Type::quadrilateral;
  else if (type == "hexahedron")
    return Type::hexahedron;
  else
  {
    log::dolfin_error("CellType.cpp", "convert string to cell type",
                      "Unknown cell type (\"%s\")", type.c_str());
  }

  return Type::interval;
}
//-----------------------------------------------------------------------------
std::string CellType::type2string(Type type)
{
  switch (type)
  {
  case Type::point:
    return "point";
  case Type::interval:
    return "interval";
  case Type::triangle:
    return "triangle";
  case Type::tetrahedron:
    return "tetrahedron";
  case Type::quadrilateral:
    return "quadrilateral";
  case Type::hexahedron:
    return "hexahedron";
  default:
    log::dolfin_error("CellType.cpp", "convert cell type to string",
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
double CellType::h(const MeshEntity& entity) const
{
  // Get mesh geometry
  const MeshGeometry& geometry = entity.mesh().geometry();

  // Get number of cell vertices
  const int num_vertices = entity.num_entities(0);

  // Get the coordinates (Points) of the vertices
  const std::int32_t* vertices = entity.entities(0);
  dolfin_assert(vertices);
  std::array<geometry::Point, 8> points;
  dolfin_assert(num_vertices <= 8);
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
double CellType::inradius(const Cell& cell) const
{
  // Check cell type
  if (_cell_type != Type::interval && _cell_type != Type::triangle
      && _cell_type != Type::tetrahedron)
  {
    log::dolfin_error("Cell.h", "compute cell inradius",
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
  return d * V / A;
}
//-----------------------------------------------------------------------------
double CellType::radius_ratio(const Cell& cell) const
{
  const double r = inradius(cell);

  // Handle degenerate case
  if (r == 0.0)
    return 0.0;
  else
    return dim() * r / circumradius(cell);
}
//-----------------------------------------------------------------------------
bool CellType::ordered(
    const Cell& cell,
    const std::vector<std::int64_t>& local_to_global_vertex_indices) const
{
  // Get mesh topology
  const MeshTopology& topology = cell.mesh().topology();
  const std::size_t dim = topology.dim();
  const std::size_t c = cell.index();

  // Get vertices
  const std::size_t num_vertices = topology(dim, 0).size(c);
  const std::int32_t* vertices = topology(dim, 0)(c);
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
    const std::int32_t* entities = topology(dim, d)(c);

    // Iterate over entities
    for (std::size_t e = 1; e < num_entities; e++)
    {
      // Get vertices for first entity
      const std::size_t e0 = entities[e - 1];
      const std::size_t n0 = connectivity.size(e0);
      const std::int32_t* v0 = connectivity(e0);

      // Get vertices for second entity
      const std::size_t e1 = entities[e];
      const std::size_t n1 = connectivity.size(e1);
      const std::int32_t* v1 = connectivity(e1);

      // Check ordering of entities
      if (!increasing(n0, v0, n1, v1, num_vertices, vertices,
                      local_to_global_vertex_indices))
        return false;
    }
  }

  return true;
}
//-----------------------------------------------------------------------------
void CellType::sort_entities(
    std::size_t num_vertices, std::int32_t* local_vertices,
    const std::vector<std::int64_t>& local_to_global_vertex_indices)
{
  // Two cases here, either sort vertices directly (when running in
  // serial) or sort based on the global indices (when running in
  // parallel)

  // Comparison operator for sorting based on global indices
  class GlobalSort
  {
  public:
    GlobalSort(const std::vector<std::int64_t>& local_to_global_vertex_indices)
        : g(local_to_global_vertex_indices)
    {
    }

    bool operator()(const std::size_t& l, const std::size_t& r)
    {
      return g[l] < g[r];
    }

    const std::vector<std::int64_t>& g;
  };

  // Sort on global vertex indices
  GlobalSort global_sort(local_to_global_vertex_indices);
  std::sort(local_vertices, local_vertices + num_vertices, global_sort);
}
//-----------------------------------------------------------------------------
bool CellType::increasing(
    std::size_t num_vertices, const std::int32_t* local_vertices,
    const std::vector<std::int64_t>& local_to_global_vertex_indices)
{
  // Two cases here, either check vertices directly (when running in serial)
  // or check based on the global indices (when running in parallel)

  for (std::size_t v = 1; v < num_vertices; v++)
    if (local_to_global_vertex_indices[local_vertices[v - 1]]
        >= local_to_global_vertex_indices[local_vertices[v]])
      return false;
  return true;
}
//-----------------------------------------------------------------------------
bool CellType::increasing(
    std::size_t n0, const std::int32_t* v0, std::size_t n1,
    const std::int32_t* v1, std::size_t num_vertices,
    const std::int32_t* local_vertices,
    const std::vector<std::int64_t>& local_to_global_vertex_indices)
{
  dolfin_assert(n0 == n1);
  dolfin_assert(num_vertices > n0);
  const std::size_t num_non_incident = num_vertices - n0;

  // Compute non-incident vertices for first entity
  std::vector<std::size_t> w0(num_non_incident);
  std::size_t k = 0;
  for (std::size_t i = 0; i < num_vertices; i++)
  {
    const std::int32_t v = local_vertices[i];
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
    const std::int32_t v = local_vertices[i];
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
