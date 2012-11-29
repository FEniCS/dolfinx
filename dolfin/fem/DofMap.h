// Copyright (C) 2007-2012 Anders Logg and Garth N. Wells
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
// Modified by Joachim B Haga, 2012
// Modified by Mikael Mortensen, 2012
//
// First added:  2007-03-01
// Last changed: 2012-11-05

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

  class GenericVector;
  class UFC;
  class UFCMesh;
  class Restriction;

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
    DofMap(boost::shared_ptr<const ufc::dofmap> ufc_dofmap,
           const Mesh& mesh);

    /// Create restricted dof map on mesh (data is not shared)
    ///
    /// *Arguments*
    ///     ufc_dofmap (ufc::dofmap)
    ///         The ufc::dofmap.
    ///     restriction (_Restriction_)
    ///         The restriction.
    DofMap(boost::shared_ptr<const ufc::dofmap> ufc_dofmap,
           boost::shared_ptr<const Restriction> restriction);

  private:

    // Create a sub-dofmap (a view) from parent_dofmap
    DofMap(const DofMap& parent_dofmap, const std::vector<uint>& component,
           const Mesh& mesh, bool distributed);

    // Create a collapsed dofmap from parent_dofmap
    DofMap(boost::unordered_map<std::size_t, std::size_t>& collapsed_map,
           const DofMap& dofmap_view, const Mesh& mesh, bool distributed);

    /// Copy constructor
    ///
    /// *Arguments*
    ///     dofmap (_DofMap_)
    ///         The object to be copied.
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

    /// True iff dof map is restricted
    ///
    /// *Returns*
    ///     bool
    ///         True iff dof map is restricted
    bool is_restricted() const
    { return static_cast<bool>(_restriction); }

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
    ///     cell_index (std::size_t)
    ///         Index of cell
    ///
    /// *Returns*
    ///     unsigned int
    ///         Dimension of the local finite element function space.
    unsigned int cell_dimension(std::size_t cell_index) const;

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

    /// Restriction if any. If the dofmap is not restricted, a null
    /// pointer is returned.
    ///
    /// *Returns*
    ///     boost::shared_ptr<const Restriction>
    //          The restriction.
    boost::shared_ptr<const Restriction> restriction() const;

    /// Return the ownership range (dofs in this range are owned by
    /// this process)
    ///
    /// *Returns*
    ///     std::pair<unsigned int, unsigned int>
    ///         The ownership range.
    std::pair<std::size_t, std::size_t> ownership_range() const;

    /// Return map from nonlocal dofs that appear in local dof map to
    /// owning process
    ///
    /// *Returns*
    ///     boost::unordered_map<unsigned int, unsigned int>
    ///         The map from non-local dofs.
    const boost::unordered_map<std::size_t, uint>& off_process_owner() const;

    /// Return map from all shared dofs to the processes (not including the current
    /// process) that share it.
    ///
    /// *Returns*
    ///     boost::unordered_map<std::size_t, std::vector<std::size_t> >
    ///         The map from dofs to list of processes
    const boost::unordered_map<std::size_t, std::vector<std::size_t> >& shared_dofs() const;

    /// Return set of all neighbouring processes.
    ///
    /// *Returns*
    ///     std::set<uint>
    ///         The set of processes
    const std::set<uint>& neighbours() const;

    /// Local-to-global mapping of dofs on a cell
    ///
    /// *Arguments*
    ///     cell_index (std::size_t)
    ///         The cell index.
    ///
    /// *Returns*
    ///     std::vector<std::size_t>
    ///         Local-to-global mapping of dofs.
    const std::vector<DolfinIndex>& cell_dofs(std::size_t cell_index) const
    {
      dolfin_assert(cell_index < _dofmap.size());
      return _dofmap[cell_index];
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
    /// *Returns*
    ///     DofMap
    ///         The Dofmap copy.
    boost::shared_ptr<GenericDofMap> copy() const;

    /// Create a copy of the dof map
    ///
    /// *Arguments*
    ///     new_mesh (_Mesh_)
    ///         The new mesh to build the dof map on.
    ///
    /// *Returns*
    ///     DofMap
    ///         The new Dofmap copy.
    boost::shared_ptr<GenericDofMap> build(const Mesh& new_mesh) const;


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
    ///     collapsed_map (boost::unordered_map<std::size_t, std::size_t>)
    ///         The "collapsed" map.
    ///     mesh (_Mesh_)
    ///         The mesh.
    ///
    /// *Returns*
    ///     DofMap
    ///         The collapsed dofmap.
    DofMap* collapse(boost::unordered_map<std::size_t, std::size_t>& collapsed_map,
                     const Mesh& mesh) const;

    /// Set dof entries in vector to a specified value. Parallel layout
    /// of vector must be consistent with dof map range.
    ///
    /// *Arguments*
    ///     vector (_GenericVector_)
    ///         The vector to set.
    ///     value (double)
    ///         The value to set.
    void set(GenericVector& x, double value) const;

    /// Set dof entries in vector to the x[i] coordinate of the dof
    /// spatial coordinate. Parallel layout of vector must be consistent
    /// with dof map range.
    ///
    /// *Arguments*
    ///     vector (_GenericVector_)
    ///         The vector to set.
    ///     value (double)
    ///         The value to multiply to coordinate by.
    ///     component (uint)
    ///         The coordinate index.
    ///     mesh (_Mesh_)
    ///         The mesh.
    void set_x(GenericVector& x, double value, uint component,
               const Mesh& mesh) const;

    /// Return the set of dof indices
    ///
    /// *Returns*
    ///     boost::unordered_set<dolfin::std::size_t>
    ///         The set of dof indices.
    boost::unordered_set<std::size_t> dofs() const;

    /// Return the underlying dof map data. Intended for internal library
    /// use only.
    ///
    /// *Returns*
    ///     std::vector<std::vector<dolfin::std::size_t> >
    ///         The local-to-global map for each cell.
    const std::vector<std::vector<DolfinIndex> >& data() const
    { return _dofmap; }

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
                                            std::size_t& offset,
                                            const std::vector<uint>& component,
                                            const ufc::mesh ufc_mesh,
                                            const Mesh& dolfin_mesh);

    // Initialize the UFC dofmap
    static void init_ufc_dofmap(ufc::dofmap& dofmap, const ufc::mesh ufc_mesh,
                                const Mesh& dolfin_mesh);

    // Initialize UFC dofmap for restricted space
    static void init_ufc_dofmap(ufc::dofmap& dofmap,
                                const ufc::mesh ufc_mesh,
                                const Mesh& dolfin_mesh,
                                const MeshFunction<uint>& domain_markers,
                                uint domain);

    // Check dimensional consistency between UFC dofmap and the mesh
    static void check_dimensional_consistency(const ufc::dofmap& dofmap,
                                              const Mesh& mesh);

    // Local-to-global dof map (dofs for cell dofmap[i])
    std::vector<std::vector<DolfinIndex> > _dofmap;

    // UFC dof map
    boost::scoped_ptr<ufc::dofmap> _ufc_dofmap;

    // Map from UFC dof numbering to renumbered dof (ufc_dof, actual_dof)
    boost::unordered_map<std::size_t, std::size_t> ufc_map_to_dofmap;

    // Restriction, pointer zero if not restricted
    boost::shared_ptr<const Restriction> _restriction;

    // Global dimension. Note that this may differ from the global dimension
    // of the UFC dofmap if the function space is restricted or periodic.
    uint _global_dimension;

    // UFC dof map offset
    std::size_t _ufc_offset;

    // Ownership range (dofs in this range are owned by this process)
    std::pair<std::size_t, std::size_t> _ownership_range;

    // Owner (process) of dofs in local dof map that do not belong to
    // this process
    boost::unordered_map<std::size_t, uint> _off_process_owner;

    // List of processes that share a given dof
    boost::unordered_map<std::size_t, std::vector<std::size_t> > _shared_dofs;

    // Neighbours (processes that we share dofs with)
    std::set<uint> _neighbours;

    // True iff sub-dofmap (a view, i.e. not collapsed)
    bool _is_view;

    // True iff running in parallel
    bool _distributed;
    
    // Map from slave dofs to master dofs using UFC numbering
    std::map<std::size_t, std::size_t> _slave_master_map;
    
    // Map of processes that share master dofs
    std::map<std::size_t, boost::unordered_set<uint> > _master_processes;
    
  };
}

#endif
