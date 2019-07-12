// Copyright (C) 2006-2017 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "CellType.h"
#include <Eigen/Dense>
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

class IntervalCell : public mesh::CellTypeOld
{
public:
  /// Specify cell type and facet type
  IntervalCell() : mesh::CellTypeOld(CellType::interval, CellType::point) {}

  /// Return number of vertices for entity of given topological
  /// dimension
  int num_vertices(int dim) const;

  /// Create entities e of given topological dimension from vertices v
  void create_entities(Eigen::Array<std::int32_t, Eigen::Dynamic,
                                    Eigen::Dynamic, Eigen::RowMajor>& e,
                       std::size_t dim, const std::int32_t* v) const;

  /// Compute circumradius of interval
  double circumradius(const MeshEntity& interval) const;

  /// Compute squared distance to given point (3D enabled)
  double squared_distance(const mesh::Cell& cell,
                          const Eigen::Vector3d& point) const;

  /// Compute squared distance to given point. This version takes
  /// the two vertex coordinates as 3D points. This makes it
  /// possible to reuse this function for computing the (squared)
  /// distance to a triangle.
  static double squared_distance(const Eigen::Vector3d& point,
                                 const Eigen::Vector3d& a,
                                 const Eigen::Vector3d& b);

  /// Compute component i of normal of given facet with respect to
  /// the cell
  double normal(const mesh::Cell& cell, std::size_t facet, std::size_t i) const;

  /// Compute of given facet with respect to the cell
  Eigen::Vector3d normal(const mesh::Cell& cell, std::size_t facet) const;

  /// Compute normal to given cell (viewed as embedded in 2D)
  Eigen::Vector3d cell_normal(const mesh::Cell& cell) const;

  /// Compute the area/length of given facet with respect to the cell
  double facet_area(const mesh::Cell& cell, std::size_t facet) const;
};
} // namespace mesh
} // namespace dolfin
