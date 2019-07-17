// Copyright (C) 2006-2015 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "CellType.h"
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
  Cell(const Mesh& mesh, std::size_t index)
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
  CellType type() const { return _mesh->type().type; }

  /// Return number of vertices of cell
  std::size_t num_vertices() const
  {
    return mesh::num_cell_vertices(_mesh->type().type);
  }

  /// Compute (generalized) volume of cell
  ///
  /// @return     double
  ///         The volume of the cell.
  ///
  /// @code{.cpp}
  ///
  ///         UnitSquare mesh(1, 1);
  ///         Cell cell(mesh, 0);
  ///         log::info("%g", cell.volume());
  ///
  /// @endcode
  double volume() const { return mesh::volume(*this); }

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

    // return _mesh->type().circumradius(*this);
  }

  /// Compute inradius of cell
  ///
  /// @return     double
  ///         Radius of the sphere inscribed in the cell.
  ///
  /// @code{.cpp}
  ///
  ///         UnitSquareMesh mesh(1, 1);
  ///         Cell cell(mesh, 0);
  ///         log::info("%g", cell.inradius());
  ///
  /// @endcode
  double inradius() const
  {
    // We would need facet areas
    const int dim = mesh::cell_dim(_mesh->type().type);
    _mesh->create_entities(dim - 1);

    Eigen::ArrayXi cells(1);
    cells[0] = this->index();
    return mesh::inradius(this->mesh(), cells)[0];
  }

  /// Compute squared distance to given point.
  ///
  /// @param     point
  ///         The point.
  /// @return     double
  ///         The squared distance to the point.
  double squared_distance(const Eigen::Vector3d& point) const
  {
    return _mesh->type().squared_distance(*this, point);
  }

  /// Note: This is a (likely temporary) replacement for ufc::cell::local_facet
  /// Local facet index, used typically in eval functions
  mutable int local_facet;
};
} // namespace mesh
} // namespace dolfin
