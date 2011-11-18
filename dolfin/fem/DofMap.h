// Copyright (C) 2007-2011 Anders Logg and Garth N. Wells
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
// Modified by Martin Alnes, 2008
// Modified by Kent-Andre Mardal, 2009
// Modified by Ola Skavhaug, 2009
//
// First added:  2007-03-01
// Last changed: 2011-10-31

#ifndef __DOLFIN_DOF_MAP_H
#define __DOLFIN_DOF_MAP_H

#include <map>
#include <memory>
#include <utility>
#include <vector>
#include <boost/multi_array.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/unordered_map.hpp>
#include <ufc.h>

#include <dolfin/common/types.h>
#include <dolfin/mesh/Cell.h>
#include "GenericDofMap.h"

namespace dolfin
{

  class UFC;
  class UFCMesh;

  /// This class handles the mapping of degrees of freedom. It builds
  /// a dof map based on a ufc::dofmap on a specific mesh. It will
  /// reorder the dofs when running in parallel. Sub-dofmaps, both
  /// views and copies, are supported.

  class DofMap : public GenericDofMap
  {
  public:

    /// Create dof map on mesh (data is not shared)
    ///
    /// *Arguments*
    ///     ufc_dofmap (ufc::dofmap)
    ///         The ufc::dofmap.
    ///     mesh (_Mesh_)
    ///         The mesh.
    DofMap(boost::shared_ptr<const ufc::dofmap> ufc_dofmap, Mesh& mesh);

    /// Create dof map on mesh ((data is not shared), const mesh
    /// version)
    ///
    /// *Arguments*
    ///     ufc_dofmap (ufc::dofmap)
    ///         The ufc::dofmap.
    ///     mesh (_Mesh_)
    ///         The mesh.
    DofMap(boost::shared_ptr<const ufc::dofmap> ufc_dofmap, const Mesh& mesh);

    /// Copy constructor
    ///
    /// *Arguments*
    ///     dofmap (_DofMap_)
    ///         The object to be copied.
    DofMap(const DofMap& dofmap);

  private:

    /// Create a sub-dofmap (a view) from parent_dofmap
    DofMap(const DofMap& parent_dofmap, const std::vector<uint>& component,
           const Mesh& mesh, bool distributed);

    /// Create a collapsed dofmap from parent_dofmap
    DofMap(boost::unordered_map<uint, uint>& collapsed_map,
           const DofMap& dofmap_view, const Mesh& mesh, bool distributed);

  public:

    /// Destructor
    ~DofMap();

    /// True if dof map is a view into another map
    ///
    /// *Returns*
    ///     bool
    ///         True if the dof map is a sub-dof map (a view into
    ///         another map).
    bool is_view() const
    { return _is_view; }

    /// Return true iff mesh entities of topological dimension d are
    /// needed
    ///
    /// *Arguments*
    ///     d (unsigned int)
    ///         Topological dimension.
    ///
    /// *Returns*
    ///     bool
    ///         True if the mesh entities are needed.
    bool needs_mesh_entities(unsigned int d) const;

    /// Return the dimension of the global finite element function
    /// space
    ///
    /// *Returns*
    ///     unsigned int
    ///         The dimension of the global finite element function space.
    unsigned int global_dimension() const;

    // FIXME: Rename this function, 'cell_dimension' sounds confusing

    /// Return the dimension of the local finite element function
    /// space on a cell
    ///
    /// *Arguments*
    ///     cell_index (uint)
    ///         Index of cell
    ///
    /// *Returns*
    ///     unsigned int
    ///         Dimension of the local finite element function space.
    unsigned int cell_dimension(uint cell_index) const;

    /// Return the maximum dimension of the local finite element
    /// function space
    ///
    /// *Returns*
    ///     unsigned int
    ///         Maximum dimension of the local finite element function
    ///         space.
    unsigned int max_cell_dimension() const;

    /// Return the geometric dimension of the coordinates this dof map
    /// provides
    ///
    /// *Returns*
    ///     unsigned int
    ///         The geometric dimension.
    unsigned int geometric_dimension() const;

    /// Return number of facet dofs
    ///
    /// *Returns*
    ///     unsigned int
    ///         The number of facet dofs.
    unsigned int num_facet_dofs() const;

    /// Return the ownership range (dofs in this range are owned by
    /// this process)
    ///
    /// *Returns*
    ///     std::pair<unsigned int, unsigned int>
    ///         The ownership range.
    std::pair<unsigned int, unsigned int> ownership_range() const;

    /// Return map from nonlocal dofs that appear in local dof map to
    /// owning process
    ///
    /// *Returns*
    ///     boost::unordered_map<unsigned int, unsigned int>
    ///         The map from non-local dofs.
    const boost::unordered_map<unsigned int, unsigned int>& off_process_owner() const;

    /// Local-to-global mapping of dofs on a cell
    ///
    /// *Arguments*
    ///     cell_index (uint)
    ///         The cell index.
    ///
    /// *Returns*
    ///     std::vector<uint>
    ///         Local-to-global mapping of dofs.
    const std::vector<uint>& cell_dofs(uint cell_index) const
    {
      dolfin_assert(cell_index < _dofmap.size());
      return _dofmap[cell_index];
    }

    /// Tabulate the local-to-global mapping of dofs on a cell
    ///
    /// *Arguments*
    ///     dofs (uint)
    ///         Degrees of freedom on a cell.
    ///     cell (_Cell_)
    ///         The cell.
    void tabulate_dofs(uint* dofs, const Cell& cell) const
    {
      const uint cell_index = cell.index();
      dolfin_assert(cell_index < _dofmap.size());
      std::copy(_dofmap[cell_index].begin(), _dofmap[cell_index].end(), dofs);
    }

    /// Tabulate local-local facet dofs
    ///
    /// *Arguments*
    ///     dofs (uint)
    ///         Degrees of freedom.
    ///     local_facet (uint)
    ///         The local facet.
    void tabulate_facet_dofs(uint* dofs, uint local_facet) const;

    /// Tabulate the coordinates of all dofs on a cell (UFC cell
    /// version)
    ///
    /// *Arguments*
    ///     coordinates (boost::multi_array<double, 2>)
    ///         The coordinates of all dofs on a cell.
    ///     ufc_cell (ufc::cell)
    ///         The cell.
    void tabulate_coordinates(boost::multi_array<double, 2>& coordinates,
                                      const ufc::cell& ufc_cell) const;

    /// Tabulate the coordinates of all dofs on a cell (DOLFIN cell
    /// version)
    ///
    /// *Arguments*
    ///     coordinates (boost::multi_array<double, 2>)
    ///         The coordinates of all dofs on a cell.
    ///     cell (_Cell_)
    ///         The cell.
    void tabulate_coordinates(boost::multi_array<double, 2>& coordinates,
                                      const Cell& cell) const;

    /// Create a copy of the dof map
    ///
    /// *Arguments*
    ///     mesh (_Mesh_)
    ///         The object to be copied.
    DofMap* copy(const Mesh& mesh) const;

    /// Extract subdofmap component
    ///
    /// *Arguments*
    ///     component (std::vector<uint>)
    ///         The component.
    ///     mesh (_Mesh_)
    ///         The mesh.
    ///
    /// *Returns*
    ///     DofMap
    ///         The subdofmap component.
    DofMap* extract_sub_dofmap(const std::vector<uint>& component,
                               const Mesh& mesh) const;

    /// Create a "collapsed" dofmap (collapses a sub-dofmap)
    ///
    /// *Arguments*
    ///     collapsed_map (boost::unordered_map<uint, uint>)
    ///         The "collapsed" map.
    ///     mesh (_Mesh_)
    ///         The mesh.
    ///
    /// *Returns*
    ///     DofMap
    ///         The collapsed dofmap.
    DofMap* collapse(boost::unordered_map<uint, uint>& collapsed_map,
                     const Mesh& mesh) const;

    /// Return the set of dof indices
    ///
    /// *Returns*
    ///     boost::unordered_set<dolfin::uint>
    ///         The set of dof indices.
    boost::unordered_set<uint> dofs() const;

    /// Return the underlying dof map data. Intended for internal library
    /// use only.
    ///
    /// *Returns*
    ///     std::vector<std::vector<dolfin::uint> >
    ///         The local-to-global map for each cell.
    const std::vector<std::vector<uint> >& data() const
    { return _dofmap; }

    /// Renumber dofs
    ///
    /// *Arguments*
    ///     renumbering_map (std::vector<uint>)
    ///         The map of dofs to be renumbered.
    void renumber(const std::vector<uint>& renumbering_map);

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

    // Recursively extract UFC sub-dofmap and compute offset
    static ufc::dofmap* extract_ufc_sub_dofmap(const ufc::dofmap& ufc_dofmap,
                                            uint& offset,
                                            const std::vector<uint>& component,
                                            const ufc::mesh ufc_mesh,
                                            const Mesh& dolfin_mesh);

    // Initialize the UFC dofmap
    static void init_ufc_dofmap(ufc::dofmap& dofmap, const ufc::mesh ufc_mesh,
                                const Mesh& dolfin_mesh);

    // Check dimensional consistency between UFC dofmap and the mesh
    static void check_dimensional_consistency(const ufc::dofmap& dofmap,
                                              const Mesh& mesh);

    // Local-to-global dof map (dofs for cell dofmap[i])
    std::vector<std::vector<uint> > _dofmap;

    // UFC dof map
    boost::scoped_ptr<ufc::dofmap> _ufc_dofmap;

    // Map from UFC dof numbering to renumbered dof (ufc_dof, actual_dof)
    boost::unordered_map<uint, uint> ufc_map_to_dofmap;

    // UFC dof map offset
    unsigned int ufc_offset;

    // Ownership range (dofs in this range are owned by this process)
    std::pair<uint, uint> _ownership_range;

    // Owner (process) of dofs in local dof map that do not belong to
    // this process
    boost::unordered_map<uint, uint> _off_process_owner;

    // True iff sub-dofmap (a view, i.e. not collapsed)
    bool _is_view;

    // True iff running in parallel
    bool _distributed;

  };

}

#endif
