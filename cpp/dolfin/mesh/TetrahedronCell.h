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
class Cell;

/// This class implements functionality for tetrahedral cell meshes.

class TetrahedronCell : public CellTypeOld
{
public:
  /// Specify cell type and facet type
  TetrahedronCell()
      : mesh::CellTypeOld(CellType::tetrahedron, CellType::triangle)
  {
  }

  /// Return number of vertices for entity of given topological dimension
  int num_vertices(int dim) const;

  /// Create entities e of given topological dimension from vertices v
  void create_entities(Eigen::Array<std::int32_t, Eigen::Dynamic,
                                    Eigen::Dynamic, Eigen::RowMajor>& e,
                       std::size_t dim, const std::int32_t* v) const;

  /// Compute circumradius of tetrahedron
  double circumradius(const MeshEntity& tetrahedron) const;

  /// Compute squared distance to given point
  double squared_distance(const mesh::Cell& cell,
                          const Eigen::Vector3d& point) const;

  /// Compute component i of normal of given facet with respect to
  /// the cell
  double normal(const mesh::Cell& cell, std::size_t facet, std::size_t i) const;

  /// Compute normal of given facet with respect to the cell
  Eigen::Vector3d normal(const mesh::Cell& cell, std::size_t facet) const;

  /// Compute normal to given cell (viewed as embedded in 4D ...)
  Eigen::Vector3d cell_normal(const mesh::Cell& cell) const;

  /// Compute the area/length of given facet with respect to the cell
  double facet_area(const mesh::Cell& cell, std::size_t facet) const;

private:
  // Find local index of edge i according to ordering convention
  std::size_t find_edge(std::size_t i, const mesh::Cell& cell) const;

  // Check whether point is outside region defined by facet ABC.
  // The fourth vertex is needed to define the orientation.
  bool point_outside_of_plane(const Eigen::Vector3d& point,
                              const Eigen::Vector3d& A,
                              const Eigen::Vector3d& B,
                              const Eigen::Vector3d& C,
                              const Eigen::Vector3d& D) const;
};
} // namespace mesh
} // namespace dolfin
