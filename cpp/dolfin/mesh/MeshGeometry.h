// Copyright (C) 2006 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfin/geometry/Point.h>
#include <dolfin/log/log.h>
#include <string>
#include <vector>

namespace dolfin
{
namespace mesh
{

/// MeshGeometry stores the geometry imposed on a mesh.

/// Currently, the geometry is represented by the set of coordinates for the
/// vertices of a mesh, but other representations are possible.
class MeshGeometry
{
public:
  /// Create empty set of coordinates
  MeshGeometry();

  /// Copy constructor
  MeshGeometry(const MeshGeometry&) = default;

  /// Move constructor
  MeshGeometry(MeshGeometry&&) = default;

  /// Destructor
  ~MeshGeometry() = default;

  /// Assignment
  MeshGeometry& operator=(const MeshGeometry&) = default;

  /// Return Euclidean dimension of coordinate system
  std::size_t dim() const { return _dim; }

  /// Return polynomial degree of coordinate field
  std::size_t degree() const { return _degree; }

  /// Return the number of vertex coordinates
  std::size_t num_vertices() const
  {
    dolfin_assert(coordinates.size() % _dim == 0);
    if (_degree > 1)
    {
      dolfin_assert(entity_offsets.size() > 1);
      dolfin_assert(entity_offsets[1].size() > 0);
      return entity_offsets[1][0];
    }
    return coordinates.size() / _dim;
  }

  /// Return the total number of points in the geometry, located on
  /// any entity
  std::size_t num_points() const
  {
    dolfin_assert(coordinates.size() % _dim == 0);
    return coordinates.size() / _dim;
  }

  /// Get vertex coordinates
  const double* vertex_coordinates(std::size_t point_index)
  {
    dolfin_assert(point_index < num_vertices());
    return &coordinates[point_index * _dim];
  }

  /// Get vertex coordinates
  const double* point_coordinates(std::size_t point_index)
  {
    dolfin_assert(point_index * _dim < coordinates.size());
    return &coordinates[point_index * _dim];
  }

  /// Return value of coordinate with local index n in direction i
  double x(std::size_t n, std::size_t i) const
  {
    dolfin_assert((n * _dim + i) < coordinates.size());
    dolfin_assert(i < _dim);
    return coordinates[n * _dim + i];
  }

  /// Return array of values for coordinate with local index n
  const double* x(std::size_t n) const
  {
    dolfin_assert(n * _dim < coordinates.size());
    return &coordinates[n * _dim];
  }

  /// Return array of values for all coordinates
  std::vector<double>& x() { return coordinates; }

  /// Return array of values for all coordinates
  const std::vector<double>& x() const { return coordinates; }

  /// Return coordinate with local index n as a 3D point value
  geometry::Point point(std::size_t n) const;

  /// Initialize coordinate list to given dimension and degree
  void init(std::size_t dim, std::size_t degree);

  /// Initialise entities. To be called after init
  void init_entities(const std::vector<std::size_t>& num_entities);

  /// Get the number of coordinate points per entity for this degree
  std::size_t num_entity_coordinates(std::size_t entity_dim) const
  {
    // Calculate the number of points per entity for Lagrange
    // elements
    switch (entity_dim)
    {
    case 0:
      return 1;
    case 1:
      return (_degree - 1);
    case 2:
      return (_degree - 2) * (_degree - 1) / 2;
    case 3:
      return (_degree - 3) * (_degree - 2) * (_degree - 1) / 6;
    }
    log::dolfin_error("MeshGeometry.h", "calculate number of points",
                      "Entity dimension out of range");
    return 0;
  }

  /// Get the index for an entity point in coordinates
  std::size_t get_entity_index(std::size_t entity_dim, std::size_t order,
                               std::size_t index) const
  {
    dolfin_assert(entity_dim < entity_offsets.size());
    dolfin_assert(order < entity_offsets[entity_dim].size());
    const std::size_t idx = (entity_offsets[entity_dim][order] + index);
    dolfin_assert(idx * _dim < coordinates.size());
    return idx;
  }

  /// Set value of coordinate
  void set(std::size_t local_index, const double* x);

  /// Hash of coordinate values
  ///
  /// @returns std::size_t
  ///    A tree-hashed value of the coordinates over all MPI processes
  ///
  std::size_t hash() const;

  /// Return informal string representation (pretty-print)
  std::string str(bool verbose) const;

private:
  // Euclidean dimension
  std::size_t _dim;

  // Polynomial degree (1 = linear, 2 = quadratic etc.)
  std::size_t _degree;

  // Offsets to storage for coordinate points for each entity type
  std::vector<std::vector<std::size_t>> entity_offsets;

  // Coordinates for all points stored as a contiguous array
  std::vector<double> coordinates;
};
}
}
