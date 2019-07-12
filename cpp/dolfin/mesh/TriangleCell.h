// Copyright (C) 2006-2017 Anders Logg
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

/// This class implements functionality for triangular meshes.

class TriangleCell : public CellTypeOld
{
public:
  /// Specify cell type and facet type
  TriangleCell() : CellTypeOld(CellType::triangle, CellType::interval) {}

  /// Return number of vertices for entity of given topological dimension
  int num_vertices(int dim) const;

  /// Create entities e of given topological dimension from vertices v
  void create_entities(Eigen::Array<std::int32_t, Eigen::Dynamic,
                                    Eigen::Dynamic, Eigen::RowMajor>& e,
                       std::size_t dim, const std::int32_t* v) const;

  /// Compute (generalized) volume (area) of triangle
  double volume(const MeshEntity& triangle) const;

  /// Compute diameter of triangle
  double circumradius(const MeshEntity& triangle) const;

  /// Compute squared distance to given point (3D enabled)
  double squared_distance(const mesh::Cell& cell,
                          const Eigen::Vector3d& point) const;

  /// Compute squared distance to given point. This version takes
  /// the three vertex coordinates as 3D points. This makes it
  /// possible to reuse this function for computing the (squared)
  /// distance to a tetrahedron.
  static double squared_distance(const Eigen::Vector3d& point,
                                 const Eigen::Vector3d& a,
                                 const Eigen::Vector3d& b,
                                 const Eigen::Vector3d& c);

  /// Compute component i of normal of given facet with respect to the cell
  double normal(const mesh::Cell& cell, std::size_t facet, std::size_t i) const;

  /// Compute of given facet with respect to the cell
  Eigen::Vector3d normal(const mesh::Cell& cell, std::size_t facet) const;

  /// Compute normal to given cell (viewed as embedded in 3D)
  Eigen::Vector3d cell_normal(const mesh::Cell& cell) const;

  /// Compute the area/length of given facet with respect to the cell
  double facet_area(const mesh::Cell& cell, std::size_t facet) const;

private:
  // Find local index of edge i according to ordering convention
  std::size_t find_edge(std::size_t i, const mesh::Cell& cell) const;
};
} // namespace mesh
} // namespace dolfin
