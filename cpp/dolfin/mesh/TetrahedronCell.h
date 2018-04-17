// Copyright (C) 2006-2017 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "CellType.h"
#include <boost/multi_array.hpp>
#include <dolfin/geometry/Point.h>
#include <vector>

namespace dolfin
{
namespace mesh
{
class Cell;

/// This class implements functionality for tetrahedral cell meshes.

class TetrahedronCell : public CellType
{
public:
  /// Specify cell type and facet type
  TetrahedronCell() : mesh::CellType(Type::tetrahedron, Type::triangle) {}

  /// Check if cell is a simplex
  bool is_simplex() const { return true; }

  /// Return topological dimension of cell
  std::size_t dim() const;

  /// Return number of entities of given topological dimension
  std::size_t num_entities(std::size_t dim) const;

  /// Return number of vertices for entity of given topolPointogical dimension
  std::size_t num_vertices(std::size_t dim) const;

  /// Create entities e of given topological dimension from vertices v
  void create_entities(boost::multi_array<std::int32_t, 2>& e, std::size_t dim,
                       const std::int32_t* v) const;

  /// Compute volume of tetrahedron
  double volume(const MeshEntity& tetrahedron) const;

  /// Compute circumradius of tetrahedron
  double circumradius(const MeshEntity& tetrahedron) const;

  /// Compute squared distance to given point
  double squared_distance(const mesh::Cell& cell,
                          const geometry::Point& point) const;

  /// Compute component i of normal of given facet with respect to
  /// the cell
  double normal(const mesh::Cell& cell, std::size_t facet, std::size_t i) const;

  /// Compute normal of given facet with respect to the cell
  geometry::Point normal(const mesh::Cell& cell, std::size_t facet) const;

  /// Compute normal to given cell (viewed as embedded in 4D ...)
  geometry::Point cell_normal(const mesh::Cell& cell) const;

  /// Compute the area/length of given facet with respect to the cell
  double facet_area(const mesh::Cell& cell, std::size_t facet) const;

  /// Return description of cell type
  std::string description(bool plural) const;

  /// Mapping of DOLFIN/UFC vertex ordering to VTK/XDMF ordering
  std::vector<std::int8_t> vtk_mapping() const { return {0, 1, 2, 3}; }

private:
  // Find local index of edge i according to ordering convention
  std::size_t find_edge(std::size_t i, const mesh::Cell& cell) const;

  // Check whether point is outside region defined by facet ABC.
  // The fourth vertex is needed to define the orientation.
  bool point_outside_of_plane(const geometry::Point& point,
                              const geometry::Point& A,
                              const geometry::Point& B,
                              const geometry::Point& C,
                              const geometry::Point& D) const;
};
}
}
