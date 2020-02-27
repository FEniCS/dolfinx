// Copyright (C) 2006-2018 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
#include <dolfinx/graph/AdjacencyList.h>
#include <memory>
#include <string>
#include <vector>

namespace dolfinx
{
namespace common
{
class IndexMap;
}

namespace fem
{
class CoordinateElement;
}

namespace mesh
{

/// Geometry stores the geometry imposed on a mesh.
///
/// Currently, the geometry is represented by the set of coordinates for
/// the vertices of a mesh, but other representations are possible.

class Geometry
{
public:
  /// Constructor (new)
  Geometry(std::shared_ptr<const common::IndexMap> index_map,
           const graph::AdjacencyList<std::int32_t>& dofmap,
           const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                              Eigen::RowMajor>& x,
           const std::vector<std::int64_t>& global_indices, int degree);

  /// Constructor
  Geometry(std::int64_t num_points_global,
           const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                              Eigen::RowMajor>& coordinates,
           const std::vector<std::int64_t>& global_indices,
           const Eigen::Ref<const Eigen::Array<
               std::int32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>&
               point_dofs,
           int degree);

  /// Copy constructor
  Geometry(const Geometry&) = default;

  /// Move constructor
  Geometry(Geometry&&) = default;

  /// Destructor
  ~Geometry() = default;

  /// Copy Assignment
  Geometry& operator=(const Geometry&) = delete;

  /// Move Assignment
  Geometry& operator=(Geometry&&) = default;

  /// Return Euclidean dimension of coordinate system
  int dim() const;

  /// @todo Remove this. Just here for mesh::Ordering
  ///
  /// DOF map
  graph::AdjacencyList<std::int32_t>& dofmap();

  /// DOF map
  const graph::AdjacencyList<std::int32_t>& dofmap() const;

  /// Geometry degrees-of-freedom
  Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>& x();

  /// Geometry degrees-of-freedom
  const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>& x() const;

  /// Return coordinate array for point with local index n
  Eigen::Ref<const Eigen::Vector3d> x(int n) const;

  /// @todo Remove this once all Geometry objects have an IndexMap
  ///
  /// Return the number of global points in the geometry
  std::size_t num_points_global() const;

  /// Global indices for points (const)
  const std::vector<std::int64_t>& global_indices() const;

  /// Hash of coordinate values
  /// @return A tree-hashed value of the coordinates over all MPI
  ///         processes
  std::size_t hash() const;

  /// Return informal string representation (pretty-print)
  std::string str(bool verbose) const;

  /// @warning Experimental
  ///
  /// Put CoordinateElement for now
  std::shared_ptr<const fem::CoordinateElement> coord_mapping;

  ///  @todo Remove this
  ///
  /// Polynomial degree of the mesh geometry
  int degree() const;

private:
  // Coordinates for all points stored as a contiguous array
  Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor> _coordinates;

  // Geometric dimension
  int _dim;

  // Global indices for points
  std::vector<std::int64_t> _global_indices;

  // Global number of points (taking account of shared points)
  std::uint64_t _num_points_global;

  // NEW
  std::shared_ptr<const common::IndexMap> _index_map;
  graph::AdjacencyList<std::int32_t> _dofmap;
  // Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor> _x;

  // FIXME: Remove this Mesh geometric degree (in Lagrange basis)
  // describing coordinate dofs
  std::int32_t _degree;
};
} // namespace mesh
} // namespace dolfinx
