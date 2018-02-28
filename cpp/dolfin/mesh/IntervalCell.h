// Copyright (C) 2006-2017 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "CellType.h"
#include <boost/multi_array.hpp>
#include <string>
#include <vector>

namespace dolfin
{

namespace mesh
{
class Cell;
class MeshEntity;
template <typename T>
class MeshFunction;

/// This class implements functionality for interval cell meshes.

class IntervalCell : public mesh::CellType
{
public:
  /// Specify cell type and facet type
  IntervalCell() : mesh::CellType(Type::interval, Type::point) {}

  /// Check if cell is a simplex
  bool is_simplex() const { return true; }

  /// Return topological dimension of cell
  std::size_t dim() const;

  /// Return number of entities of given topological dimension
  std::size_t num_entities(std::size_t dim) const;

  /// Return number of vertices for entity of given topological
  /// dimension
  std::size_t num_vertices(std::size_t dim) const;

  /// Create entities e of given topological dimension from vertices v
  void create_entities(boost::multi_array<std::int32_t, 2>& e, std::size_t dim,
                       const std::int32_t* v) const;

  /// Compute (generalized) volume (length) of interval
  double volume(const MeshEntity& interval) const;

  /// Compute circumradius of interval
  double circumradius(const MeshEntity& interval) const;

  /// Compute squared distance to given point (3D enabled)
  double squared_distance(const mesh::Cell& cell, const Point& point) const;

  /// Compute squared distance to given point. This version takes
  /// the two vertex coordinates as 3D points. This makes it
  /// possible to reuse this function for computing the (squared)
  /// distance to a triangle.
  static double squared_distance(const Point& point, const Point& a,
                                 const Point& b);

  /// Compute component i of normal of given facet with respect to
  /// the cell
  double normal(const mesh::Cell& cell, std::size_t facet, std::size_t i) const;

  /// Compute of given facet with respect to the cell
  Point normal(const mesh::Cell& cell, std::size_t facet) const;

  /// Compute normal to given cell (viewed as embedded in 2D)
  Point cell_normal(const mesh::Cell& cell) const;

  /// Compute the area/length of given facet with respect to the cell
  double facet_area(const mesh::Cell& cell, std::size_t facet) const;

  /// Order entities locally
  void
  order(mesh::Cell& cell,
        const std::vector<std::int64_t>& local_to_global_vertex_indices) const;

  /// Return description of cell type
  std::string description(bool plural) const;

  /// Mapping of DOLFIN/UFC vertex ordering to VTK/XDMF ordering
  std::vector<std::int8_t> vtk_mapping() const { return {0, 1}; }
};
}
}
