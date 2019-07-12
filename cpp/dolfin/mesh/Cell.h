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
  double volume() const { return _mesh->type().volume(*this); }

  /// Compute greatest distance between any two vertices
  ///
  /// @return     double
  ///         The greatest distance between any two vertices of the cell.
  ///
  /// @code{.cpp}
  ///
  ///         UnitSquareMesh mesh(1, 1);
  ///         Cell cell(mesh, 0);
  ///         log::info("%g", cell.h());
  ///
  /// @endcode
  double h() const { return _mesh->type().h(*this); }

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
  double circumradius() const { return _mesh->type().circumradius(*this); }

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
    _mesh->create_entities(_mesh->type().dim() - 1);

    return _mesh->type().inradius(*this);
  }

  /// Compute ratio of inradius to circumradius times dim for cell.
  /// Useful as cell quality measure. Returns 1. for equilateral
  /// and 0. for degenerate cell.
  /// See Jonathan Richard Shewchuk: What Is a Good Linear Finite Element?,
  /// online: http://www.cs.berkeley.edu/~jrs/papers/elemj.pdf
  ///
  /// @return     double
  ///         topological_dimension * inradius / circumradius
  ///
  /// @code{.cpp}
  ///
  ///         UnitSquareMesh mesh(1, 1);
  ///         Cell cell(mesh, 0);
  ///         log::info("%g", cell.radius_ratio());
  ///
  /// @endcode
  double radius_ratio() const
  {
    // We would need facet areas
    _mesh->create_entities(_mesh->type().dim() - 1);

    return _mesh->type().radius_ratio(*this);
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

  /// Compute distance to given point.
  ///
  ///  @param    point
  ///         The point.
  /// @return     double
  ///         The distance to the point.
  double distance(const Eigen::Vector3d& point) const
  {
    return sqrt(squared_distance(point));
  }

  /// Compute normal of given facet with respect to the cell
  ///
  /// @param    facet
  ///         Index of facet.
  ///
  /// @return Eigen::Vector3d
  ///         Normal of the facet.
  Eigen::Vector3d normal(std::size_t facet) const
  {
    return _mesh->type().normal(*this, facet);
  }

  /// Compute normal to cell itself (viewed as embedded in 3D)
  ///
  /// @return geometry::Point
  ///         Normal of the cell
  Eigen::Vector3d cell_normal() const
  {
    return _mesh->type().cell_normal(*this);
  }

  /// Compute the area/length of given facet with respect to the cell
  ///
  /// @param    facet
  ///         Index of the facet.
  ///
  /// @return     double
  ///         Area/length of the facet.
  double facet_area(std::size_t facet) const
  {
    return _mesh->type().facet_area(*this, facet);
  }

  /// Note: This is a (likely temporary) replacement for ufc::cell::local_facet
  /// Local facet index, used typically in eval functions
  mutable int local_facet;
};
} // namespace mesh
} // namespace dolfin
