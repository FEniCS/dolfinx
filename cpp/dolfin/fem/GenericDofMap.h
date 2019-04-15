// Copyright (C) 2010-2015 Anders Logg and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
#include <memory>
#include <petscsys.h>
#include <petscvec.h>
#include <set>
#include <unordered_map>
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
class SubDomain;
} // namespace mesh

namespace fem
{

/// This class provides a generic interface for dof maps

class GenericDofMap
{
public:
  /// Destructor
  virtual ~GenericDofMap() = default;

  /// True if dof map is a view into another map (is a sub-dofmap)
  virtual bool is_view() const = 0;

  /// Return the dimension of the global finite element function
  /// space
  virtual std::int64_t global_dimension() const = 0;

  /// Return the dimension of the local finite element function
  /// space on a cell
  virtual std::size_t num_element_dofs(std::size_t index) const = 0;

  /// Return the maximum dimension of the local finite element
  /// function space
  virtual std::size_t max_element_dofs() const = 0;

  /// Return the number of dofs for a given entity dimension
  virtual std::size_t num_entity_dofs(std::size_t entity_dim) const = 0;

  /// Return the number of closure dofs for a given entity dimension
  virtual std::size_t num_entity_closure_dofs(std::size_t entity_dim) const = 0;

  /// Return the ownership range (dofs in this range are owned by
  /// this process)
  virtual std::array<std::int64_t, 2> ownership_range() const = 0;

  /// Local-to-global mapping of dofs on a cell
  virtual Eigen::Map<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>>
  cell_dofs(std::size_t cell_index) const = 0;

  /// Return the dof indices associated with entities of given dimension and
  /// entity indices
  Eigen::Array<PetscInt, Eigen::Dynamic, 1>
  entity_dofs(const mesh::Mesh& mesh, std::size_t entity_dim,
              const std::vector<std::size_t>& entity_indices) const;

  /// Return the dof indices associated with all entities of given dimension
  Eigen::Array<PetscInt, Eigen::Dynamic, 1>
  entity_dofs(const mesh::Mesh& mesh, std::size_t entity_dim) const;

  /// Tabulate the local-to-local mapping of dofs on entity
  /// (dim, local_entity)
  virtual Eigen::Array<int, Eigen::Dynamic, 1>
  tabulate_entity_dofs(std::size_t entity_dim,
                       std::size_t cell_entity_index) const = 0;

  /// Tabulate local-local closure dofs on entity
  virtual Eigen::Array<int, Eigen::Dynamic, 1>
  tabulate_entity_closure_dofs(std::size_t entity_dim,
                               std::size_t cell_entity_index) const = 0;

  /// Tabulate globally supported dofs
  virtual Eigen::Array<std::size_t, Eigen::Dynamic, 1>
  tabulate_global_dofs() const = 0;

  /// Extract sub dofmap component
  virtual std::unique_ptr<GenericDofMap>
  extract_sub_dofmap(const std::vector<std::size_t>& component,
                     const mesh::Mesh& mesh) const = 0;

  /// Create a "collapsed" a dofmap (collapses from a sub-dofmap view)
  virtual std::pair<std::shared_ptr<GenericDofMap>,
                    std::unordered_map<std::size_t, std::size_t>>
  collapse(const mesh::Mesh& mesh) const = 0;

  /// Return list of dof indices on this process that belong to mesh
  /// entities of dimension dim
  Eigen::Array<PetscInt, Eigen::Dynamic, 1> dofs(const mesh::Mesh& mesh,
                                                 std::size_t dim) const;

  /// Set dof entries in vector to a specified value. Parallel
  /// layout of vector must be consistent with dof map range. This
  /// function is typically used to construct the null space of a
  /// matrix operator
  virtual void set(Vec x, PetscScalar value) const = 0;

  /// Index map (const access)
  virtual std::shared_ptr<const common::IndexMap> index_map() const = 0;

  /// Tabulate map between local (process) and global dof indices
  Eigen::Array<std::size_t, Eigen::Dynamic, 1>
  tabulate_local_to_global_dofs() const;

  /// Return map from shared nodes to the processes (not including
  /// the current process) that share it.
  virtual const std::unordered_map<int, std::vector<int>>&
  shared_nodes() const = 0;

  /// Return set of processes that share dofs with the this process
  virtual const std::set<int>& neighbours() const = 0;

  /// Return informal string representation (pretty-print)
  virtual std::string str(bool verbose) const = 0;

  /// Get block size
  virtual int block_size() const = 0;
};
} // namespace fem
} // namespace dolfin
