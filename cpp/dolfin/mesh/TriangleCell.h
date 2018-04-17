// Copyright (C) 2006-2017 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "CellType.h"
#include <boost/multi_array.hpp>
#include <vector>

namespace dolfin
{
namespace mesh
{

/// This class implements functionality for triangular meshes.

class TriangleCell : public CellType
{
public:
  /// Specify cell type and facet type
  TriangleCell() : CellType(Type::triangle, Type::interval) {}

  /// Check if cell is a simplex
  bool is_simplex() const { return true; }

  /// Return topological dimension of cell
  std::size_t dim() const;

  /// Return number of entities of given topological dimension
  std::size_t num_entities(std::size_t dim) const;

  /// Return number of vertices for entity of given topological dimension
  std::size_t num_vertices(std::size_t dim) const;

  /// Create entities e of given topological dimension from vertices v
  void create_entities(boost::multi_array<std::int32_t, 2>& e, std::size_t dim,
                       const std::int32_t* v) const;

  /// Compute (generalized) volume (area) of triangle
  double volume(const MeshEntity& triangle) const;

  /// Compute diameter of triangle
  double circumradius(const MeshEntity& triangle) const;

  /// Compute squared distance to given point (3D enabled)
  double squared_distance(const mesh::Cell& cell,
                          const geometry::Point& point) const;

  /// Compute squared distance to given point. This version takes
  /// the three vertex coordinates as 3D points. This makes it
  /// possible to reuse this function for computing the (squared)
  /// distance to a tetrahedron.
  static double squared_distance(const geometry::Point& point,
                                 const geometry::Point& a,
                                 const geometry::Point& b,
                                 const geometry::Point& c);

  /// Compute component i of normal of given facet with respect to the cell
  double normal(const mesh::Cell& cell, std::size_t facet, std::size_t i) const;

  /// Compute of given facet with respect to the cell
  geometry::Point normal(const mesh::Cell& cell, std::size_t facet) const;

  /// Compute normal to given cell (viewed as embedded in 3D)
  geometry::Point cell_normal(const mesh::Cell& cell) const;

  /// Compute the area/length of given facet with respect to the cell
  double facet_area(const mesh::Cell& cell, std::size_t facet) const;

  /// Return description of cell type
  std::string description(bool plural) const;

  /// Mapping of DOLFIN/UFC vertex ordering to VTK/XDMF ordering
  std::vector<std::int8_t> vtk_mapping() const { return {0, 1, 2}; }

private:
  // Find local index of edge i according to ordering convention
  std::size_t find_edge(std::size_t i, const mesh::Cell& cell) const;
};
}
}
