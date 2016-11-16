// Copyright (C) 2007-2015 Anders Logg and Garth N. Wells
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Martin Alnes, 2008-2015
// Modified by Kent-Andre Mardal, 2009
// Modified by Ola Skavhaug, 2009
// Modified by Joachim B Haga, 2012
// Modified by Mikael Mortensen, 2012
// Modified by Jan Blechta, 2013

#ifndef __DOLFIN_DOF_MAP_H
#define __DOLFIN_DOF_MAP_H

#include <cstdlib>
#include <map>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>
#include <ufc.h>

#include <dolfin/common/ArrayView.h>
#include <dolfin/common/types.h>
#include <dolfin/la/IndexMap.h>
#include <dolfin/mesh/Cell.h>
#include "GenericDofMap.h"

namespace dolfin
{

  class GenericVector;

  /// This class handles the mapping of degrees of freedom. It builds
  /// a dof map based on a ufc::dofmap on a specific mesh. It will
  /// reorder the dofs when running in parallel. Sub-dofmaps, both
  /// views and copies, are supported.

  class DofMap : public GenericDofMap
  {
  public:

    /// Create dof map on mesh (mesh is not stored)
    ///
    /// *Arguments*
    ///     ufc_dofmap (ufc::dofmap)
    ///         The ufc::dofmap.
    ///     mesh (_Mesh_)
    ///         The mesh.
    DofMap(std::shared_ptr<const ufc::dofmap> ufc_dofmap,
           const Mesh& mesh);

    /// Create a periodic dof map on mesh (mesh is not stored)
    ///
    /// *Arguments*
    ///     ufc_dofmap (ufc::dofmap)
    ///         The ufc::dofmap.
    ///     mesh (_Mesh_)
    ///         The mesh.
    ///     constrained_boundary (_SubDomain_)
    ///         The subdomain marking the constrained (tied) boundaries.
    DofMap(std::shared_ptr<const ufc::dofmap> ufc_dofmap,
           const Mesh& mesh, std::shared_ptr<const SubDomain> constrained_domain);

  private:

    // Create a sub-dofmap (a view) from parent_dofmap
    DofMap(const DofMap& parent_dofmap,
           const std::vector<std::size_t>& component,
           const Mesh& mesh);

    // Create a collapsed dofmap from parent_dofmap
    DofMap(std::unordered_map<std::size_t, std::size_t>& collapsed_map,
           const DofMap& dofmap_view, const Mesh& mesh);

    // Copy constructor
    DofMap(const DofMap& dofmap);

  public:

    /// Destructor
    ~DofMap();

    /// True iff dof map is a view into another map
    ///
    /// *Returns*
    ///     bool
    ///         True if the dof map is a sub-dof map (a view into
    ///         another map).
    bool is_view() const
    { return _is_view; }

    /// Return the dimension of the global finite element function
    /// space
    ///
    /// *Returns*
    ///     std::size_t
    ///         The dimension of the global finite element function space.
    std::size_t global_dimension() const;

    /// Return the dimension of the local finite element function
    /// space on a cell
    ///
    /// *Arguments*
    ///     cell_index (std::size_t)
    ///         Index of cell
    ///
    /// *Returns*
    ///     std::size_t
    ///         Dimension of the local finite element function space.
    std::size_t num_element_dofs(std::size_t cell_index) const;

    /// Return the maximum dimension of the local finite element
    /// function space
    ///
    /// *Returns*
    ///     std::size_t
    ///         Maximum dimension of the local finite element function
    ///         space.
    std::size_t max_element_dofs() const;

    /// Return the number of dofs for a given entity dimension
    ///
    /// *Arguments*
    ///     dim (std::size_t)
    ///         Entity dimension
    ///
    /// *Returns*
    ///     std::size_t
    ///         Number of dofs associated with given entity dimension
    virtual std::size_t num_entity_dofs(std::size_t dim) const;

    /// Return number of facet dofs
    ///
    /// *Returns*
    ///     std::size_t
    ///         The number of facet dofs.
    std::size_t num_facet_dofs() const;

    /// Return the ownership range (dofs in this range are owned by
    /// this process)
    ///
    /// *Returns*
    ///     std::pair<std::size_t, std::size_t>
    ///         The ownership range.
    std::pair<std::size_t, std::size_t> ownership_range() const;

    /// Return map from nonlocal dofs that appear in local dof map to
    /// owning process
    ///
    /// *Returns*
    ///     std::vector<unsigned int>
    ///         The map from non-local dofs.
    const std::vector<int>& off_process_owner() const
    { return _index_map->off_process_owner(); }

    /// Return map from all shared nodes to the sharing processes (not
    /// including the current process) that share it.
    ///
    /// *Returns*
    ///     std::unordered_map<std::size_t, std::vector<unsigned int>>
    ///         The map from dofs to list of processes
    const std::unordered_map<int, std::vector<int>>& shared_nodes() const;

    /// Return set of processes that share dofs with this process
    ///
    /// *Returns*
    ///     std::set<int>
    ///         The set of processes
    const std::set<int>& neighbours() const;

    /// Clear any data required to build sub-dofmaps (this is to
    /// reduce memory use)
    void clear_sub_map_data()
    {
      //std::vector<int>().swap(_ufc_local_to_local);
      _ufc_local_to_local.clear();
    }

    /// Local-to-global mapping of dofs on a cell
    ///
    /// *Arguments*
    ///     cell_index (std::size_t)
    ///         The cell index.
    ///
    /// *Returns*
    ///     ArrayView<const dolfin::la_index>
    ///         Local-to-global mapping of dofs.
    ArrayView<const dolfin::la_index> cell_dofs(std::size_t cell_index) const
    {
      const std::size_t index = cell_index*_cell_dimension;
      dolfin_assert(index + _cell_dimension <= _dofmap.size());
      return ArrayView<const dolfin::la_index>(_cell_dimension,
                                               &_dofmap[index]);
    }

    /// Tabulate local-local facet dofs
    ///
    /// *Arguments*
    ///     dofs (std::size_t)
    ///         Degrees of freedom.
    ///     local_facet (std::size_t)
    ///         The local facet.
    void tabulate_facet_dofs(std::vector<std::size_t>& dofs,
                             std::size_t local_facet) const;

    /// Tabulate local-local mapping of dofs on entity (dim, local_entity)
    ///
    /// *Arguments*
    ///     dofs (std::size_t)
    ///         Degrees of freedom.
    ///     dim (std::size_t)
    ///         The entity dimension
    ///     local_entity (std::size_t)
    ///         The local entity index
    void tabulate_entity_dofs(std::vector<std::size_t>& dofs,
                              std::size_t dim, std::size_t local_entity) const;

    /// Tabulate globally supported dofs
    ///
    /// *Arguments*
    ///     dofs (std::size_t)
    ///         Degrees of freedom.
    void tabulate_global_dofs(std::vector<std::size_t>& dofs) const
    {
      dolfin_assert(_global_nodes.empty() || block_size() == 1);
      dofs.resize(_global_nodes.size());
      std::copy(_global_nodes.cbegin(), _global_nodes.cend(), dofs.begin());
    }

    /// Create a copy of the dof map
    ///
    /// *Returns*
    ///     DofMap
    ///         The Dofmap copy.
    std::shared_ptr<GenericDofMap> copy() const;

    /// Create a copy of the dof map on a new mesh
    ///
    /// *Arguments*
    ///     new_mesh (_Mesh_)
    ///         The new mesh to create the dof map on.
    ///
    /// *Returns*
    ///     DofMap
    ///         The new Dofmap copy.
    std::shared_ptr<GenericDofMap> create(const Mesh& new_mesh) const;


    /// Extract subdofmap component
    ///
    /// *Arguments*
    ///     component (std::vector<std::size_t>)
    ///         The component.
    ///     mesh (_Mesh_)
    ///         The mesh.
    ///
    /// *Returns*
    ///     DofMap
    ///         The subdofmap component.
    std::shared_ptr<GenericDofMap>
      extract_sub_dofmap(const std::vector<std::size_t>& component,
                         const Mesh& mesh) const;

    /// Create a "collapsed" dofmap (collapses a sub-dofmap)
    ///
    /// *Arguments*
    ///     collapsed_map (std::unordered_map<std::size_t, std::size_t>)
    ///         The "collapsed" map.
    ///     mesh (_Mesh_)
    ///         The mesh.
    ///
    /// *Returns*
    ///     DofMap
    ///         The collapsed dofmap.
    std::shared_ptr<GenericDofMap>
      collapse(std::unordered_map<std::size_t, std::size_t>&
               collapsed_map, const Mesh& mesh) const;

    // FIXME: Document this function properly
    /// Return list of dof indices on this process that belong to mesh
    /// entities of dimension dim
    std::vector<dolfin::la_index> dofs(const Mesh& mesh,
                                       std::size_t dim) const;

    // FIXME: Document this function
    std::vector<dolfin::la_index> dofs() const;

    /// Set dof entries in vector to a specified value. Parallel layout
    /// of vector must be consistent with dof map range. This
    /// function is typically used to construct the null space of a
    /// matrix operator.
    ///
    /// *Arguments*
    ///     vector (_GenericVector_)
    ///         The vector to set.
    ///     value (double)
    ///         The value to set.
    void set(GenericVector& x, double value) const;

    /// Return the map (const access)
    std::shared_ptr<const IndexMap> index_map() const
    { return _index_map; }

    /// Return the block size for dof maps with components, typically
    /// used for vector valued functions.
    int block_size() const
    { return _index_map->block_size(); }

    /// Compute the map from local (this process) dof indices to
    /// global dof indices.
    ///
    /// *Arguments*
    ///     local_to_global_map (_std::vector<std::size_t>_)
    ///         The local-to-global map to fill.
    void tabulate_local_to_global_dofs(std::vector<std::size_t>& local_to_global_map) const;

    /// Return global dof index for a given local (process) dof index
    ///
    /// *Arguments*
    ///     local_index (int)
    ///         The local local index.
    ///
    /// *Returns*
    ///     std::size_t
    ///         The global dof index.
    std::size_t local_to_global_index(int local_index) const
    { return _index_map->local_to_global(local_index); }

    /// Return indices of dofs which are owned by other processes
    const std::vector<std::size_t>& local_to_global_unowned() const
    { return _index_map->local_to_global_unowned(); }

    /// Return informal string representation (pretty-print)
    ///
    /// *Arguments*
    ///     verbose (bool)
    ///         Flag to turn on additional output.
    ///
    /// *Returns*
    ///     std::string
    ///         An informal representation of the function space.
    std::string str(bool verbose) const;

  private:

    // Friends
    friend class DofMapBuilder;

    // Check dimensional consistency between UFC dofmap and the mesh
    static void check_dimensional_consistency(const ufc::dofmap& dofmap,
                                              const Mesh& mesh);

    // Check that mesh provides the entities needed by dofmap
    static void check_provided_entities(const ufc::dofmap& dofmap,
                                        const Mesh& mesh);

    // Cell-local-to-dof map (dofs for cell dofmap[i])
    std::vector<dolfin::la_index> _dofmap;

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
    std::size_t _global_dimension;

    // UFC dof map offset
    std::size_t _ufc_offset;

    // Multimesh dof map offset
    std::size_t _multimesh_offset;

    // Object containing information about dof distribution across
    // processes
    std::shared_ptr<IndexMap> _index_map;

    // Temporary until MultiMeshDofMap runs in parallel
    friend class MultiMeshDofMap;

    // List of processes that share a given dof
    std::unordered_map<int, std::vector<int>> _shared_nodes;

    // Neighbours (processes that we share dofs with)
    std::set<int> _neighbours;

  };
}

#endif
