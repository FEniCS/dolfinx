// Copyright (C) 2007-2018 Anders Logg and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "GenericDofMap.h"
#include "petscsys.h"
#include <Eigen/Dense>
#include <array>
#include <cstdlib>
#include <memory>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

struct ufc_dofmap;

namespace dolfin
{

namespace common
{
class IndexMap;
}

namespace la
{
class PETScVector;
}

namespace mesh
{
class Mesh;
class SubDomain;
} // namespace mesh

namespace fem
{

/// Degree-of-freedom map

/// This class handles the mapping of degrees of freedom. It builds a
/// dof map based on a ufc_dofmap on a specific mesh. It will reorder
/// the dofs when running in parallel. Sub-dofmaps, both views and
/// copies, are supported.

class DofMap : public GenericDofMap
{
public:
  /// Create dof map on mesh (mesh is not stored)
  ///
  /// @param[in] ufc_dofmap (ufc_dofmap)
  ///         The ufc_dofmap.
  /// @param[in] mesh (mesh::Mesh&)
  ///         The mesh.
  DofMap(std::shared_ptr<const ufc_dofmap> ufc_dofmap, const mesh::Mesh& mesh);

private:
  // Create a sub-dofmap (a view) from parent_dofmap
  DofMap(const DofMap& parent_dofmap, const std::vector<std::size_t>& component,
         const mesh::Mesh& mesh);

  // Create a collapsed dofmap from parent_dofmap
  DofMap(std::unordered_map<std::size_t, std::size_t>& collapsed_map,
         const DofMap& dofmap_view, const mesh::Mesh& mesh);

public:
  // Copy constructor
  DofMap(const DofMap& dofmap) = delete;

  /// Move constructor
  DofMap(DofMap&& dofmap) = default;

  /// Destructor
  ~DofMap() = default;

  DofMap& operator=(const DofMap& dofmap) = delete;

  /// Move assignment
  DofMap& operator=(DofMap&& dofmap) = default;

  /// True iff dof map is a view into another map
  ///
  /// @returns bool
  ///         True if the dof map is a sub-dof map (a view into
  ///         another map).
  bool is_view() const { return _ufc_offset >= 0; }

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

  /// Return the ownership range (dofs in this range are owned by
  /// this process)
  ///
  /// @return   std::array<std::size_t, 2>
  ///         The ownership range.
  std::array<std::int64_t, 2> ownership_range() const;

  /// Return map from all shared nodes to the sharing processes (not
  /// including the current process) that share it.
  ///
  /// @return     std::unordered_map<std::size_t, std::vector<std::uint32_t>>
  ///         The map from dofs to list of processes
  const std::unordered_map<int, std::vector<int>>& shared_nodes() const;

  /// Return set of processes that share dofs with this process
  ///
  /// @return     std::set<int>
  ///         The set of processes
  const std::set<int>& neighbours() const;

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
    const std::size_t index = cell_index * _cell_dimension;
    assert(index + _cell_dimension <= _dofmap.size());
    return Eigen::Map<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>>(
        &_dofmap[index], _cell_dimension);
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

  /// Tabulate globally supported dofs
  Eigen::Array<std::size_t, Eigen::Dynamic, 1> tabulate_global_dofs() const;

  /// Extract subdofmap component
  ///
  /// @param     component (std::vector<std::size_t>)
  ///         The component.
  /// @param     mesh (_mesh::Mesh_)
  ///         The mesh.
  ///
  /// @return     DofMap
  ///         The subdofmap component.
  std::unique_ptr<GenericDofMap>
  extract_sub_dofmap(const std::vector<std::size_t>& component,
                     const mesh::Mesh& mesh) const;

  /// Create a "collapsed" dofmap (collapses a sub-dofmap)
  ///
  /// @param     collapsed_map (std::unordered_map<std::size_t, std::size_t>)
  ///         The "collapsed" map.
  /// @param     mesh (_mesh::Mesh_)
  ///         The mesh.
  ///
  /// @return    DofMap
  ///         The collapsed dofmap.
  std::pair<std::shared_ptr<GenericDofMap>,
            std::unordered_map<std::size_t, std::size_t>>
  collapse(const mesh::Mesh& mesh) const;

  /// Set dof entries in vector to a specified value. Parallel layout
  /// of vector must be consistent with dof map range. This
  /// function is typically used to construct the null space of a
  /// matrix operator.
  ///
  /// @param  x (la::PETScVector)
  ///         The vector to set.
  /// @param  value (PetscScalar)
  ///         The value to set.
  void set(la::PETScVector& x, PetscScalar value) const;

  /// Return the map (const access)
  std::shared_ptr<const common::IndexMap> index_map() const;

  /// Return the block size for dof maps with components, typically
  /// used for vector valued functions.
  int block_size() const;

  /// Return informal string representation (pretty-print)
  ///
  /// @param     verbose (bool)
  ///         Flag to turn on additional output.
  ///
  /// @return    std::string
  ///         An informal representation of the function space.
  std::string str(bool verbose) const;

private:
  // Check that mesh provides the entities needed by dofmap
  static void check_provided_entities(const ufc_dofmap& dofmap,
                                      const mesh::Mesh& mesh);

  // Cell-local-to-dof map (dofs for cell dofmap[i])
  std::vector<PetscInt> _dofmap;

  // List of global nodes
  std::set<std::size_t> _global_nodes;

  // Cell dimension (fixed for all cells)
  int _cell_dimension;

  // UFC dof map
  std::shared_ptr<const ufc_dofmap> _ufc_dofmap;

  // Global dimension
  std::int64_t _global_dimension;

  // UFC dof map offset (< 0 if not a view)
  std::int64_t _ufc_offset;

  // Object containing information about dof distribution across
  // processes
  std::shared_ptr<const common::IndexMap> _index_map;

  // Processes that share a given dof
  std::unordered_map<int, std::vector<int>> _shared_nodes;

  // Processes that this dofmap shares dofs with
  std::set<int> _neighbours;
};
} // namespace fem
} // namespace dolfin
