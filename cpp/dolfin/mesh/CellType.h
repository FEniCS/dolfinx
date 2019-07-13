// Copyright (C) 2006-2017 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "utils.h"
#include <Eigen/Dense>
#include <cstdint>
#include <string>
#include <vector>

namespace dolfin
{

namespace mesh
{
class Cell;
class MeshEntity;

/// This class provides a common interface for different cell types.
/// Each cell type implements mesh functionality that is specific to
/// a certain type of cell.

class CellTypeOld
{
public:
  /// Constructor
  CellTypeOld(CellType cell_type, CellType facet_type);

  /// Destructor
  virtual ~CellTypeOld() = default;

  /// Create cell type from type (factory function)
  static CellTypeOld* create(CellType type);

  /// Return type of cell for entity of dimension d
  CellType entity_type(int d) const;

  /// Return number of vertices for entity of given topological
  /// dimension
  virtual int num_vertices(int dim) const = 0;

  /// Create entities e of given topological dimension from vertices v
  virtual void create_entities(Eigen::Array<std::int32_t, Eigen::Dynamic,
                                            Eigen::Dynamic, Eigen::RowMajor>& e,
                               std::size_t dim,
                               const std::int32_t* v) const = 0;

  /// Compute greatest distance between any two vertices
  virtual double h(const MeshEntity& entity) const;

  /// Compute circumradius of mesh entity
  virtual double circumradius(const MeshEntity& entity) const = 0;

  /// Compute inradius of cell
  virtual double inradius(const Cell& cell) const;

  /// Compute dim*inradius/circumradius for given cell
  virtual double radius_ratio(const Cell& cell) const;

  /// Compute squared distance to given point
  virtual double squared_distance(const Cell& cell,
                                  const Eigen::Vector3d& point) const = 0;

  /// Compute component i of normal of given facet with respect to the
  /// cell
  virtual double normal(const Cell& cell, std::size_t facet,
                        std::size_t i) const = 0;

  /// Compute of given facet with respect to the cell
  virtual Eigen::Vector3d normal(const Cell& cell, std::size_t facet) const = 0;

  /// Compute normal to given cell (viewed as embedded in 3D)
  virtual Eigen::Vector3d cell_normal(const Cell& cell) const = 0;

  /// Compute the area/length of given facet with respect to the cell
  virtual double facet_area(const Cell& cell, std::size_t facet) const = 0;

  const CellType type;
  const CellType facet_type;
};
} // namespace mesh
} // namespace dolfin
