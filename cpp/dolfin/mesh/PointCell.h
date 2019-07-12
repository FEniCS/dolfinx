// Copyright (C) 2007-2007 Kristian B. Oelgaard
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "CellType.h"
#include <vector>

namespace dolfin
{
namespace mesh
{
/// This class implements functionality for point cell meshes.

class PointCell : public CellTypeOld
{
public:
  /// Specify cell type and facet type
  PointCell() : CellTypeOld(CellType::point, CellType::point) {}

  /// Return topological dimension of cell
  std::size_t dim() const;

  /// Return number of entities of given topological dimension
  std::size_t num_entities(std::size_t dim) const;

  /// Return number of vertices for entity of given topological
  /// dimension
  std::size_t num_vertices(std::size_t dim) const;

  /// Create entities e of given topological dimension from vertices v
  void create_entities(Eigen::Array<std::int32_t, Eigen::Dynamic,
                                    Eigen::Dynamic, Eigen::RowMajor>& e,
                       std::size_t dim, const std::int32_t* v) const;

  /// Compute (generalized) volume (area) of triangle
  double volume(const MeshEntity& triangle) const;

  /// Compute circumradius of PointCell
  double circumradius(const MeshEntity& point) const;

  /// Compute squared distance to given point
  double squared_distance(const mesh::Cell& cell,
                          const Eigen::Vector3d& point) const;

  /// Compute component i of normal of given facet with respect to
  /// the cell
  double normal(const mesh::Cell& cell, std::size_t facet, std::size_t i) const;

  /// Compute of given facet with respect to the cell
  Eigen::Vector3d normal(const mesh::Cell& cell, std::size_t facet) const;

  /// Compute normal to given cell (viewed as embedded in 1D)
  Eigen::Vector3d cell_normal(const mesh::Cell& cell) const;

  /// Compute the area/length of given facet with respect to the
  /// cell
  double facet_area(const mesh::Cell& cell, std::size_t facet) const;

private:
  // Find local index of edge i according to ordering convention
  std::size_t find_edge(std::size_t i, const mesh::Cell& cell) const;
};
} // namespace mesh
} // namespace dolfin
