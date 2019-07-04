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
  /// Create dof map on mesh
  ///
  /// @param[in] ElementDofLayout
  ///         The layout of dofs on an element.
  /// @param[in] mesh (mesh::Mesh&)
  ///         The mesh.
  DofMap(std::shared_ptr<const ElementDofLayout> element_dof_layout,
         const mesh::Mesh& mesh);

private:
  // Create a sub-dofmap (a view) from parent_dofmap
  DofMap(const DofMap& dofmap_parent, const std::vector<int>& component,
         const mesh::Mesh& mesh);

  // Create a collapsed dofmap from parent_dofmap
  DofMap(const DofMap& dofmap_view, const mesh::Mesh& mesh);

public:
  // Copy constructor
  DofMap(const DofMap& dofmap) = delete;

  /// Move constructor
  DofMap(DofMap&& dofmap) = default;

  /// Destructor
  virtual ~DofMap() = default;

  DofMap& operator=(const DofMap& dofmap) = delete;

  /// Move assignment
  DofMap& operator=(DofMap&& dofmap) = default;

  /// True iff dof map is a view into another map
  ///
  /// @returns bool
  ///         True if the dof map is a sub-dof map (a view into
  ///         another map).
  bool is_view() const;

  /// Return the dimension of the global finite element function
  /// space. Use index_map()->size() to get the local dimension.
  ///
  /// @returns std::int64_t
  ///         The dimension of the global finite element function space.
  std::int64_t global_dimension() const;

  /// Return the dimension of the local finite element function
  /// space on a cell
  ///
  /// @param      cell_index (std::size_t)
  ///         Index of cell
  ///
  /// @return     std::size_t
  ///         Dimension of the local finite element function space.
  std::size_t num_element_dofs(std::size_t cell_index) const;

  /// Return the maximum dimension of the local finite element
  /// function space
  ///
  /// @return     std::size_t
  ///         Maximum dimension of the local finite element function
  ///         space.
  std::size_t max_element_dofs() const;

  /// Return the number of dofs for a given entity dimension
  ///
  /// @param     entity_dim (std::size_t)
  ///         Entity dimension
  ///
  /// @return     std::size_t
  ///         Number of dofs associated with given entity dimension
  virtual std::size_t num_entity_dofs(std::size_t entity_dim) const;

  /// Return the number of closure dofs for a given entity dimension
  /// s
  /// @param     entity_dim (std::size_t)
  ///         Entity dimension
  ///
  /// @return     std::size_t
  ///         Number of dofs associated with closure of given entity dimension
  virtual std::size_t num_entity_closure_dofs(std::size_t entity_dim) const;

  /// Local-to-global mapping of dofs on a cell
  ///
  /// @param     cell_index (std::size_t)
  ///         The cell index.
  ///
  /// @return         Eigen::Map<const Eigen::Array<PetscInt,
  /// Eigen::Dynamic, 1>>
  Eigen::Map<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>>
  cell_dofs(std::size_t cell_index) const
  {
    assert(_element_dof_layout);
    const int cell_dimension = _element_dof_layout->num_dofs();
    const std::size_t index = cell_index * cell_dimension;
    assert(index + cell_dimension <= _dofmap.size());
    return Eigen::Map<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>>(
        &_dofmap[index], cell_dimension);
  }

  /// Tabulate local-local closure dofs on entity of cell
  ///
  /// @param   entity_dim (std::size_t)
  ///         The entity dimension.
  /// @param    cell_entity_index (std::size_t)
  ///         The local entity index on the cell.
  /// @return     Eigen::Array<int, Eigen::Dynamic, 1>
  ///         Degrees of freedom on a single element.
  Eigen::Array<int, Eigen::Dynamic, 1>
  tabulate_entity_closure_dofs(std::size_t entity_dim,
                               std::size_t cell_entity_index) const;

  /// Tabulate local-local mapping of dofs on entity of cell
  ///
  /// @param   entity_dim (std::size_t)
  ///         The entity dimension.
  /// @param    cell_entity_index (std::size_t)
  ///         The local entity index on the cell.
  /// @return     Eigen::Array<int, Eigen::Dynamic, 1>
  ///         Degrees of freedom on a single element.
  Eigen::Array<int, Eigen::Dynamic, 1>
  tabulate_entity_dofs(std::size_t entity_dim,
                       std::size_t cell_entity_index) const;

  /// Extract subdofmap component
  ///
  /// @param     component (std::vector<int>)
  ///         The component.
  /// @param     mesh (_mesh::Mesh_)
  ///         The mesh.
  ///
  /// @return     DofMap
  ///         The subdofmap component.
  std::unique_ptr<DofMap> extract_sub_dofmap(const std::vector<int>& component,
                                             const mesh::Mesh& mesh) const;

  /// Create a "collapsed" dofmap (collapses a sub-dofmap)
  ///
  /// @param     collapsed_map
  ///         The "collapsed" map.
  /// @param     mesh (_mesh::Mesh_)
  ///         The mesh.
  ///
  /// @return    DofMap
  ///         The collapsed dofmap.
  std::pair<std::shared_ptr<DofMap>, std::vector<PetscInt>>
  collapse(const mesh::Mesh& mesh) const;

  /// Set dof entries in vector to a specified value. Parallel layout
  /// of vector must be consistent with dof map range. This
  /// function is typically used to construct the null space of a
  /// matrix operator.
  ///
  /// @param  x
  ///         The vector to set.
  /// @param  value (PetscScalar)
  ///         The value to set.
  void set(Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> x,
           PetscScalar value) const;

  /// Return the map
  std::shared_ptr<const common::IndexMap> index_map() const;

  /// Return informal string representation (pretty-print)
  ///
  /// @param     verbose (bool)
  ///         Flag to turn on additional output.
  ///
  /// @return    std::string
  ///         An informal representation of the function space.
  std::string str(bool verbose) const;

  /// Get dofmap array
  Eigen::Map<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>> dof_array() const;

  // FIXME: can this be removed?
  /// Tabulate map between local (process) and global dof indices
  Eigen::Array<std::size_t, Eigen::Dynamic, 1>
  tabulate_local_to_global_dofs() const;

  // FIXME: can this be removed?
  /// Return list of dof indices on this process that belong to mesh
  /// entities of dimension dim
  Eigen::Array<PetscInt, Eigen::Dynamic, 1> dofs(const mesh::Mesh& mesh,
                                                 std::size_t dim) const;

private:
  // Cell-local-to-dof map (dofs for cell dofmap[i])
  std::vector<PetscInt> _dofmap;

  // Object containing information about dof distribution across
  // processes
  std::shared_ptr<const common::IndexMap> _index_map;

  std::shared_ptr<const ElementDofLayout> _element_dof_layout;
};
} // namespace fem
} // namespace dolfin
