// Copyright (C) 2006-2017 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <boost/multi_array.hpp>
#include <cstdint>
#include <string>
#include <vector>

namespace dolfin
{

class MeshEntity;
template <typename T>
class MeshFunction;
class Point;

namespace mesh
{

class Cell;

/// This class provides a common interface for different cell types.
/// Each cell type implements mesh functionality that is specific to
/// a certain type of cell.

class CellType
{
public:
  /// Enum for different cell types
  enum class Type : int
  {
    point,
    interval,
    triangle,
    quadrilateral,
    tetrahedron,
    hexahedron
  };

  /// Constructor
  CellType(Type cell_type, Type facet_type);

  /// Destructor
  virtual ~CellType();

  /// Create cell type from type (factory function)
  static CellType* create(Type type);

  /// Create cell type from string (factory function)
  static CellType* create(std::string type);

  /// Convert from string to cell type
  static Type string2type(std::string type);

  /// Convert from cell type to string
  static std::string type2string(Type type);

  /// Return type of cell
  Type cell_type() const { return _cell_type; }

  /// Return type of cell for facets
  Type facet_type() const { return _facet_type; }

  /// Return type of cell for entity of dimension i
  Type entity_type(std::size_t i) const;

  /// Check if cell is a simplex
  virtual bool is_simplex() const = 0;

  /// Return topological dimension of cell
  virtual std::size_t dim() const = 0;

  /// Return number of entities of given topological dimension
  virtual std::size_t num_entities(std::size_t dim) const = 0;

  /// Return number of vertices for cell
  int num_vertices() const { return num_vertices(dim()); }

  /// Return number of vertices for entity of given topological dimension
  virtual std::size_t num_vertices(std::size_t dim) const = 0;

  /// Create entities e of given topological dimension from
  /// vertices v
  virtual void create_entities(boost::multi_array<std::int32_t, 2>& e,
                               std::size_t dim,
                               const std::int32_t* v) const = 0;

  /// Compute (generalized) volume of mesh entity
  virtual double volume(const MeshEntity& entity) const = 0;

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
                                  const Point& point) const = 0;

  /// Compute component i of normal of given facet with respect to the cell
  virtual double normal(const Cell& cell, std::size_t facet,
                        std::size_t i) const = 0;

  /// Compute of given facet with respect to the cell
  virtual Point normal(const Cell& cell, std::size_t facet) const = 0;

  /// Compute normal to given cell (viewed as embedded in 3D)
  virtual Point cell_normal(const Cell& cell) const = 0;

  /// Compute the area/length of given facet with respect to the cell
  virtual double facet_area(const Cell& cell, std::size_t facet) const = 0;

  // FIXME: The order() function should be reimplemented and use one common
  // FIXME: implementation for all cell types, just as we have for ordered()

  /// Order entities locally
  virtual void order(Cell& cell, const std::vector<std::int64_t>&
                                     local_to_global_vertex_indices) const = 0;

  /// Check if entities are ordered
  bool ordered(
      const Cell& cell,
      const std::vector<std::int64_t>& local_to_global_vertex_indices) const;

  /// Check whether given point collides with cell
  virtual bool collides(const Cell& cell, const Point& point) const = 0;

  /// Check whether given entity collides with cell
  virtual bool collides(const Cell& cell, const MeshEntity& entity) const = 0;

  /// Return description of cell type
  virtual std::string description(bool plural) const = 0;

  /// Mapping of DOLFIN/UFC vertex ordering to VTK/XDMF ordering
  virtual std::vector<std::int8_t> vtk_mapping() const = 0;

protected:
  Type _cell_type;
  Type _facet_type;

  /// Sort vertices based on global entity indices
  static void sort_entities(
      std::size_t num_vertices, std::int32_t* vertices,
      const std::vector<std::int64_t>& local_to_global_vertex_indices);

private:
  // Check if list of vertices is increasing
  static bool
  increasing(std::size_t num_vertices, const std::int32_t* vertices,
             const std::vector<std::int64_t>& local_to_global_vertex_indices);

  // Check that <entity e0 with vertices v0> <= <entity e1 with vertices v1>
  static bool
  increasing(std::size_t n0, const std::int32_t* v0, std::size_t n1,
             const std::int32_t* v1, std::size_t num_vertices,
             const std::int32_t* vertices,
             const std::vector<std::int64_t>& local_to_global_vertex_indices);
};
}
}