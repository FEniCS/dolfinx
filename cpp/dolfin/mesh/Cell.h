// Copyright (C) 2006-2015 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "CellType.h"
#include "Mesh.h"
#include "MeshEntity.h"
#include "MeshFunction.h"
#include <Eigen/Dense>
#include <dolfin/common/types.h>
#include <dolfin/geometry/Point.h>
#include <memory>
#include <ufc.h>

namespace dolfin
{
namespace mesh
{

/// A Cell is a _MeshEntity_ of topological codimension 0.

class Cell : public MeshEntity
{
public:
  // FIXME: can thos be removed?
  /// Create empty cell
  Cell() : MeshEntity() {}

  /// Create cell on given mesh with given index
  ///
  /// @param    mesh
  ///         The mesh.
  /// @param    index
  ///         The index.
  Cell(const Mesh& mesh, std::size_t index)
      : MeshEntity(mesh, mesh.topology().dim(), index)
  {
  }

  /// Destructor
  ~Cell() {}

  /// Return type of cell
  CellType::Type type() const { return _mesh->type().cell_type(); }

  /// Return number of vertices of cell
  std::size_t num_vertices() const { return _mesh->type().num_vertices(); }

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
    _mesh->init(_mesh->type().dim() - 1);

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
    _mesh->init(_mesh->type().dim() - 1);

    return _mesh->type().radius_ratio(*this);
  }

  /// Compute squared distance to given point.
  ///
  /// @param     point
  ///         The point.
  /// @return     double
  ///         The squared distance to the point.
  double squared_distance(const geometry::Point& point) const
  {
    return _mesh->type().squared_distance(*this, point);
  }

  /// Compute distance to given point.
  ///
  ///  @param    point
  ///         The point.
  /// @return     double
  ///         The distance to the point.
  double distance(const geometry::Point& point) const
  {
    return sqrt(squared_distance(point));
  }

  /// Compute normal of given facet with respect to the cell
  ///
  /// @param    facet
  ///         Index of facet.
  ///
  /// @return geometry::Point
  ///         Normal of the facet.
  geometry::Point normal(std::size_t facet) const
  {
    return _mesh->type().normal(*this, facet);
  }

  /// Compute normal to cell itself (viewed as embedded in 3D)
  ///
  /// @return geometry::Point
  ///         Normal of the cell
  geometry::Point cell_normal() const
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

  /// Order entities locally
  ///
  /// @param    local_to_global_vertex_indices
  ///         The global vertex indices.
  void order(const std::vector<std::int64_t>& local_to_global_vertex_indices)
  {
    _mesh->type().order(*this, local_to_global_vertex_indices);
  }

  /// Check if entities are ordered
  ///
  ///  @param    local_to_global_vertex_indices
  ///         The global vertex indices.
  ///
  /// @return     bool
  ///         True iff ordered.
  bool
  ordered(const std::vector<std::int64_t>& local_to_global_vertex_indices) const
  {
    return _mesh->type().ordered(*this, local_to_global_vertex_indices);
  }

  // FIXME: This function is part of a UFC transition
  /// Get cell coordinate dofs (not vertex coordinates)
  void get_coordinate_dofs(Eigen::Ref<EigenRowArrayXXd> coordinates) const
  {
    const MeshGeometry& geom = _mesh->geometry();
    const std::size_t gdim = geom.dim();
    const std::size_t geom_degree = geom.degree();
    const std::size_t num_vertices = this->num_vertices();
    const std::int32_t* vertices = this->entities(0);

    if (geom_degree == 1)
    {
      coordinates.resize(num_vertices, gdim);
      for (std::size_t i = 0; i < num_vertices; ++i)
      {
        const double* x = geom.x(vertices[i]);
        for (std::size_t j = 0; j < gdim; ++j)
          coordinates(i, j) = x[j];
      }
    }
    else
      throw std::runtime_error(
          "Cannot get coordinate_dofs. Unsupported mesh degree");
  }

  // FIXME: This function is part of a UFC transition
  /// Get cell coordinate dofs (not vertex coordinates)
  void get_coordinate_dofs(std::vector<double>& coordinates) const
  {
    const MeshGeometry& geom = _mesh->geometry();
    const std::size_t gdim = geom.dim();
    const std::size_t geom_degree = geom.degree();
    const std::size_t num_vertices = this->num_vertices();
    const std::int32_t* vertices = this->entities(0);

    if (geom_degree == 1)
    {
      coordinates.resize(num_vertices * gdim);
      for (std::size_t i = 0; i < num_vertices; ++i)
        for (std::size_t j = 0; j < gdim; ++j)
          coordinates[i * gdim + j] = geom.x(vertices[i])[j];
    }
    else if (geom_degree == 2)
    {
      const std::size_t tdim = _mesh->topology().dim();
      const std::size_t num_edges = this->num_entities(1);
      const std::int32_t* edges = this->entities(1);

      coordinates.resize((num_vertices + num_edges) * gdim);

      for (std::size_t i = 0; i < num_vertices; ++i)
        for (std::size_t j = 0; j < gdim; j++)
          coordinates[i * gdim + j] = geom.x(vertices[i])[j];

      for (std::size_t i = 0; i < num_edges; ++i)
      {
        const std::size_t entity_index = (tdim == 1) ? index() : edges[i];
        const std::size_t point_index
            = geom.get_entity_index(1, 0, entity_index);
        for (std::size_t j = 0; j < gdim; ++j)
          coordinates[(i + num_vertices) * gdim + j] = geom.x(point_index)[j];
      }
    }
    else
    {
      log::dolfin_error("Cell.h", "get coordinate_dofs",
                        "Unsupported mesh degree");
    }
  }

  // FIXME: This function is part of a UFC transition
  /// Get cell vertex coordinates (not coordinate dofs)
  void get_vertex_coordinates(std::vector<double>& coordinates) const
  {
    const std::size_t gdim = _mesh->geometry().dim();
    const std::size_t num_vertices = this->num_vertices();
    const std::int32_t* vertices = this->entities(0);
    coordinates.resize(num_vertices * gdim);
    for (std::size_t i = 0; i < num_vertices; i++)
      for (std::size_t j = 0; j < gdim; j++)
        coordinates[i * gdim + j] = _mesh->geometry().x(vertices[i])[j];
  }

  // FIXME: This function is part of a UFC transition
  /// Fill UFC cell with miscellaneous data
  void get_cell_data(ufc::cell& ufc_cell, int local_facet = -1) const
  {
    ufc_cell.geometric_dimension = _mesh->geometry().dim();
    ufc_cell.local_facet = local_facet;
    ufc_cell.orientation = -1;
    ufc_cell.mesh_identifier = this->mesh().id();
    ufc_cell.index = index();
  }
};
}
}