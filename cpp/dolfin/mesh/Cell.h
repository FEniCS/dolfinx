// Copyright (C) 2006-2015 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "CoordinateDofs.h"
#include "Geometry.h"
#include "Mesh.h"
#include "MeshEntity.h"
#include "MeshFunction.h"
#include "utils.h"
#include <Eigen/Dense>
#include <dolfin/common/types.h>
#include <memory>

namespace dolfin
{
namespace mesh
{

/// A Cell is a _MeshEntity_ of topological codimension 0.

class Cell : public MeshEntity
{
public:
  /// Create cell on given mesh with given index
  ///
  /// @param    mesh
  ///         The mesh.
  /// @param    index
  ///         The index.
  Cell(const Mesh& mesh, std::int32_t index)
      : MeshEntity(mesh, mesh.topology().dim(), index), local_facet(-1)
  {
  }

  /// Copy constructor
  Cell(const Cell& cell) = default;

  /// Move constructor
  Cell(Cell&& cell) = default;

  /// Destructor
  ~Cell() = default;

  /// Assignement operator
  Cell& operator=(const Cell& cell) = default;

  /// Return type of cell
  CellType type() const { return _mesh->cell_type; }

  /// Return number of vertices of cell
  std::size_t num_vertices() const
  {
    return mesh::num_cell_vertices(_mesh->cell_type);
  }

  /// Compute circumradius of cell
  ///
  /// @return     double
  ///         The circumradius of the cell.
  ///
  /// @code{.cpp}
  ///
  ///         UnitSquareMesh mesh(1, 1);
  ///         Cell cell(mesh, 0);
  ///         log::info("%g", cell.circumradius());
  ///
  /// @endcode
  double circumradius() const
  {
    Eigen::ArrayXi cells(1);
    cells[0] = this->index();
    return mesh::circumradius(this->mesh(), cells, this->dim())[0];
  }

  /// Note: This is a (likely temporary) replacement for ufc::cell::local_facet
  /// Local facet index, used typically in eval functions
  mutable int local_facet;
};
} // namespace mesh
} // namespace dolfin
