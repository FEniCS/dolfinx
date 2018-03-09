// Copyright (C) 2007-2015 Anders Logg and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "DofMapBuilder.h"
#include "GenericDofMap.h"
#include <Eigen/Dense>
#include <cstdlib>
#include <dolfin/common/IndexMap.h>
#include <dolfin/common/types.h>
#include <map>
#include <memory>
#include <ufc.h>
#include <unordered_map>
#include <utility>
#include <vector>

namespace dolfin
{
namespace la
{
class PETScVector;
}

namespace mesh
{
class Mesh;
class SubDomain;
}

namespace fem
{

/// Degree-of-freedom map

/// This class handles the mapping of degrees of freedom. It builds
/// a dof map based on a ufc::dofmap on a specific mesh. It will
/// reorder the dofs when running in parallel. Sub-dofmaps, both
/// views and copies, are supported.

class DofMap : public GenericDofMap
{
public:
  /// Create dof map on mesh (mesh is not stored)
  ///
  /// @param[in] ufc_dofmap (ufc::dofmap)
  ///         The ufc::dofmap.
  /// @param[in] mesh (mesh::Mesh&)
  ///         The mesh.
  DofMap(std::shared_ptr<const ufc::dofmap> ufc_dofmap, const mesh::Mesh& mesh);

  /// Create a periodic dof map on mesh (mesh is not stored)
  ///
  /// @param[in] ufc_dofmap (ufc::dofmap)
  ///         The ufc::dofmap.
  /// @param[in] mesh (mesh::Mesh)
  ///         The mesh.
  /// @param[in] constrained_domain (mesh::SubDomain)
  ///         The subdomain marking the constrained (tied) boundaries.
  DofMap(std::shared_ptr<const ufc::dofmap> ufc_dofmap, const mesh::Mesh& mesh,
         std::shared_ptr<const mesh::SubDomain> constrained_domain);

private:
  // Create a sub-dofmap (a view) from parent_dofmap
  DofMap(const DofMap& parent_dofmap, const std::vector<std::size_t>& component,
         const mesh::Mesh& mesh);

  // Create a collapsed dofmap from parent_dofmap
  DofMap(std::unordered_map<std::size_t, std::size_t>& collapsed_map,
         const DofMap& dofmap_view, const mesh::Mesh& mesh);

  // Copy constructor
  DofMap(const DofMap& dofmap) = default;

public:
  /// Move constructor
  DofMap(DofMap&& dofmap) = default;

  /// Destructor
  ~DofMap() = default;

  /// True iff dof map is a view into another map
  ///
  /// @returns bool
  ///         True if the dof map is a sub-dof map (a view into
  ///         another map).
  bool is_view() const { return _is_view; }

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

  /// Return number of facet dofs
  ///
  /// @return     std::size_t
  ///         The number of facet dofs.
  std::size_t num_facet_dofs() const;

  /// Return the ownership range (dofs in this range are owned by
  /// this process)
  ///
  /// @return   std::array<std::size_t, 2>
  ///         The ownership range.
  std::array<std::int64_t, 2> ownership_range() const;

  /// Return map from nonlocal dofs that appear in local dof map to
  /// owning process
  ///
  /// @return     std::vector<std::uint32_t>
  ///         The map from non-local dofs.
  const std::vector<int>& off_process_owner() const
  {
    return _index_map->block_off_process_owner();
  }

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
  /// @return         ArrayView<const dolfin::la_index_t>
  Eigen::Map<const Eigen::Array<dolfin::la_index_t, Eigen::Dynamic, 1>>
  cell_dofs(std::size_t cell_index) const
  {
    const std::size_t index = cell_index * _cell_dimension;
    dolfin_assert(index + _cell_dimension <= _dofmap.size());
    return Eigen::Map<const Eigen::Array<dolfin::la_index_t, Eigen::Dynamic,
                                         1>>(&_dofmap[index], _cell_dimension);
  }

  /// Tabulate local-local facet dofs
  ///
  /// @param    element_dofs (std::size_t)
  ///         Degrees of freedom on a single element.
  /// @param    cell_facet_index (std::size_t)
  ///         The local facet index on the cell.
  void tabulate_facet_dofs(std::vector<std::size_t>& element_dofs,
                           std::size_t cell_facet_index) const;

  /// Tabulate local-local mapping of dofs on entity (dim, local_entity)
  ///
  /// @param    element_dofs (std::size_t)
  ///         Degrees of freedom on a single element.
  /// @param   entity_dim (std::size_t)
  ///         The entity dimension.
  /// @param    cell_entity_index (std::size_t)
  ///         The local entity index on the cell.
  void tabulate_entity_dofs(std::vector<std::size_t>& element_dofs,
                            std::size_t entity_dim,
                            std::size_t cell_entity_index) const;


  /// Tabulate globally supported dofs
  ///
  /// @param    element_dofs (std::size_t)
  ///         Degrees of freedom.
  void tabulate_global_dofs(std::vector<std::size_t>& element_dofs) const
  {
    dolfin_assert(_global_nodes.empty() || block_size() == 1);
    element_dofs.resize(_global_nodes.size());
    std::copy(_global_nodes.cbegin(), _global_nodes.cend(),
              element_dofs.begin());
  }

  /// Extract subdofmap component
  ///
  /// @param     component (std::vector<std::size_t>)
  ///         The component.
  /// @param     mesh (_mesh::Mesh_)
  ///         The mesh.
  ///
  /// @return     DofMap
  ///         The subdofmap component.
  std::shared_ptr<GenericDofMap>
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
  std::shared_ptr<GenericDofMap>
  collapse(std::unordered_map<std::size_t, std::size_t>& collapsed_map,
           const mesh::Mesh& mesh) const;

  /// Set dof entries in vector to a specified value. Parallel layout
  /// of vector must be consistent with dof map range. This
  /// function is typically used to construct the null space of a
  /// matrix operator.
  ///
  /// @param  x (la::PETScVector)
  ///         The vector to set.
  /// @param  value (double)
  ///         The value to set.
  void set(la::PETScVector& x, double value) const;

  /// Return the map (const access)
  std::shared_ptr<const common::IndexMap> index_map() const
  {
    return _index_map;
  }

  /// Return the block size for dof maps with components, typically
  /// used for vector valued functions.
  int block_size() const { return _index_map->block_size(); }

  /// Return informal string representation (pretty-print)
  ///
  /// @param     verbose (bool)
  ///         Flag to turn on additional output.
  ///
  /// @return    std::string
  ///         An informal representation of the function space.
  std::string str(bool verbose) const;

private:
  // Friends
  friend class fem::DofMapBuilder;

  // Check that mesh provides the entities needed by dofmap
  static void check_provided_entities(const ufc::dofmap& dofmap,
                                      const mesh::Mesh& mesh);

  // Cell-local-to-dof map (dofs for cell dofmap[i])
  std::vector<dolfin::la_index_t> _dofmap;

  // List of global nodes
  std::set<std::size_t> _global_nodes;

  // Cell dimension (fixed for all cells)
  std::size_t _cell_dimension;

  // UFC dof map
  std::shared_ptr<const ufc::dofmap> _ufc_dofmap;

  // Number global mesh entities. This is usually the same as what
  // is reported by the mesh, but will differ for dofmaps
  // constrained, e.g. dofmaps with periodic bcs. It is stored in
  // order to compute the global dimension of dofmaps that are
  // constructed from a sub-dofmap.
  std::vector<std::size_t> _num_mesh_entities_global;

  // Map from UFC dof numbering to renumbered dof (ufc_dof ->
  // actual_dof, both using local indices)
  std::vector<int> _ufc_local_to_local;

  // Flag to determine if the DofMap is a view
  bool _is_view;

  // Global dimension. Note that this may differ from the global
  // dimension of the UFC dofmap if the function space is periodic.
  std::int64_t _global_dimension;

  // UFC dof map offset
  std::size_t _ufc_offset;

  // Object containing information about dof distribution across
  // processes
  std::shared_ptr<common::IndexMap> _index_map;

  // List of processes that share a given dof
  std::unordered_map<int, std::vector<int>> _shared_nodes;

  // Neighbours (processes that we share dofs with)
  std::set<int> _neighbours;
};
}
}
