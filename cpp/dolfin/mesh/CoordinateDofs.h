// Copyright (C) 2018 Chris N Richardson
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
#include <memory>
#include <vector>

namespace dolfin
{
namespace mesh
{
class Connectivity;

/// CoordinateDofs contains the connectivity from MeshEntities to the
/// geometric points which make up the mesh.

class CoordinateDofs
{
public:
  /// Constructor
  /// @param tdim
  ///   Topological dimension
  /// @param point_dofs
  ///   Array containing point dofs for each entity
  /// @param cell_permutation
  ///   Array containing permutation for cell_vertices required for higher order
  ///   elements which are input in gmsh/vtk order.
  CoordinateDofs(
      int tdim,
      const Eigen::Ref<const Eigen::Array<std::int32_t, Eigen::Dynamic,
                                          Eigen::Dynamic, Eigen::RowMajor>>
          point_dofs,
      const std::vector<std::uint8_t>& cell_permutation);

  /// Copy constructor
  CoordinateDofs(const CoordinateDofs& topology) = default;

  /// Move constructor
  CoordinateDofs(CoordinateDofs&& topology) = default;

  /// Destructor
  ~CoordinateDofs() = default;

  /// Copy assignment
  CoordinateDofs& operator=(const CoordinateDofs& topology) = default;

  /// Move assignment
  CoordinateDofs& operator=(CoordinateDofs&& topology) = default;

  /// Get the entity points associated with entities of dimension i
  ///
  /// @param dim
  ///   Entity dimension
  /// @return Connectivity
  ///   Connections from entities of given dimension to points
  const Connectivity& entity_points(std::uint32_t dim) const;

  const std::vector<std::uint8_t>& cell_permutation() const;

private:
  // Connectivity from entities to points. Initially only defined for
  // cells
  std::vector<std::shared_ptr<Connectivity>> _coord_dofs;

  // Permutation required to transform to/from VTK/gmsh ordering to
  // DOLFIN ordering needed for higher order elements
  // FIXME: ideally remove this, but would need to harmonise the dof
  // ordering between dolfin/ffc/gmsh
  std::vector<std::uint8_t> _cell_permutation;
};
} // namespace mesh
} // namespace dolfin
