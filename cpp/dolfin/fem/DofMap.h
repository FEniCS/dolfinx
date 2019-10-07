// Copyright (C) 2007-2018 Anders Logg and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "ElementDofLayout.h"
#include "petscsys.h"
#include <Eigen/Dense>
#include <array>
#include <cstdlib>
#include <memory>
#include <set>
#include <utility>
#include <vector>

namespace dolfin
{

namespace common
{
class IndexMap;
}

namespace mesh
{
class Mesh;
} // namespace mesh

namespace fem
{

/// Degree-of-freedom map

/// This class handles the mapping of degrees of freedom. It builds a
/// dof map based on an ElementDofLayout on a specific mesh. It will
/// reorder the dofs when running in parallel. Sub-dofmaps, both views
/// and copies, are supported.

class DofMap
{
public:
  /// Create a DofMap from the layout of dofs on a reference element, an
  /// IndexMap defining the distribtion of dofs across processes and a vector of
  /// indices.
  DofMap(std::shared_ptr<const ElementDofLayout> element_dof_layout,
         std::shared_ptr<const common::IndexMap> index_map,
         const Eigen::Array<PetscInt, Eigen::Dynamic, 1>& dofmap);

public:
  // Copy constructor
  DofMap(const DofMap& dofmap) = delete;

  /// Move constructor
  DofMap(DofMap&& dofmap) = default;

  /// Destructor
  virtual ~DofMap() = default;

  // Copy assignment
  DofMap& operator=(const DofMap& dofmap) = delete;

  /// Move assignment
  DofMap& operator=(DofMap&& dofmap) = default;

  /// Local-to-global mapping of dofs on a cell
  /// @param[in] cell_index The cell index.
  /// @return  Local-global map for cell (used process-local global
  ///           index)
  auto cell_dofs(int cell_index) const
  {
    assert(element_dof_layout);
    const int cell_dimension = element_dof_layout->num_dofs();
    const int index = cell_index * cell_dimension;
    assert(index + cell_dimension <= _dofmap.size());
    return _dofmap.segment(index, cell_dimension);
  }

  /// Extract subdofmap component
  /// @param[in] component The component indices
  /// @param[in] mesh The mesh the the dofmap is defined on
  /// @return The dofmap for the component
  DofMap extract_sub_dofmap(const std::vector<int>& component,
                            const mesh::Mesh& mesh) const;

  /// Create a "collapsed" dofmap (collapses a sub-dofmap)
  /// @param[in] mesh The mesh that the dofmap is defined on
  /// @return The collapsed dofmap
  std::pair<std::unique_ptr<DofMap>, std::vector<PetscInt>>
  collapse(const mesh::Mesh& mesh) const;

  /// Set dof entries in vector to a specified value. Parallel layout
  /// of vector must be consistent with dof map range. This
  /// function is typically used to construct the null space of a
  /// matrix operator.
  ///
  /// @param[in,out] x The vector to set
  /// @param[in] value The value to set on the vector
  void set(Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> x,
           PetscScalar value) const;

  /// Return informal string representation (pretty-print)
  /// @param[in] verbose Flag to turn on additional output.
  /// @return An informal representation of the function space.
  std::string str(bool verbose) const;

  /// Get dofmap array
  const Eigen::Array<PetscInt, Eigen::Dynamic, 1>& dof_array() const;

  // FIXME: can this be removed?
  /// Return list of dof indices on this process that belong to mesh
  /// entities of dimension dim
  Eigen::Array<PetscInt, Eigen::Dynamic, 1> dofs(const mesh::Mesh& mesh,
                                                 std::size_t dim) const;

  /// Layout of dofs on an element
  std::shared_ptr<const ElementDofLayout> element_dof_layout;

  /// Object containing information about dof distribution across
  /// processes
  std::shared_ptr<const common::IndexMap> index_map;

private:
  // Cell-local-to-dof map (dofs for cell dofmap[i])
  Eigen::Array<PetscInt, Eigen::Dynamic, 1> _dofmap;
};
} // namespace fem
} // namespace dolfin
