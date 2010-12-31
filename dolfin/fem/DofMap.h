// Copyright (C) 2007-2010 Anders Logg and Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Martin Alnes, 2008
// Modified by Kent-Andre Mardal, 2009
// Modified by Ola Skavhaug, 2009
//
// First added:  2007-03-01
// Last changed: 2010-06-01

#ifndef __DOF_MAP_H
#define __DOF_MAP_H

#include <map>
#include <memory>
#include <vector>
#include <boost/shared_ptr.hpp>
#include <dolfin/common/types.h>
#include "GenericDofMap.h"

namespace dolfin
{

  class UFC;
  class UFCMesh;
  //template<class T> class Set;

  /// This class handles the mapping of degrees of freedom. It builds
  /// a dof map based on a ufc::dof_map on a specific mesh. It will
  /// reorder the dofs when running in parallel.
  ///
  /// If ufc_offset != 0, then the dof map provides a view into a
  /// larger dof map. A dof map which is a view, can be 'collapsed'
  /// such that the dof indices are contiguous.

  // FIXME: Review function names for parallel dof maps

  class DofMap : public GenericDofMap
  {
  public:

    /// Create dof map on mesh
    DofMap(boost::shared_ptr<ufc::dof_map> ufc_dofmap,
           Mesh& dolfin_mesh);

    /// Create dof map on mesh (const mesh version)
    DofMap(boost::shared_ptr<ufc::dof_map> ufc_dofmap,
           const Mesh& dolfin_mesh);

  private:

    /// Create dof map on mesh with a std::vector dof map
    DofMap(boost::shared_ptr<ufc::dof_map> ufc_dofmap, const UFCMesh& ufc_mesh);

  public:

    /// Destructor
    ~DofMap();

    /// Return a string identifying the dof map
    std::string signature() const;

    /// Return true iff mesh entities of topological dimension d are needed
    bool needs_mesh_entities(unsigned int d) const;

    /// Return the dimension of the global finite element function space
    unsigned int global_dimension() const;

    /// Return the dimension of the local (process) finite element function space
    unsigned int local_dimension() const;

    /// Return the dimension of the local finite element function space on a cell
    unsigned int dimension(uint cell_index) const;

    /// Return the maximum dimension of the local finite element function space
    unsigned int max_local_dimension() const;

    // Return the geometric dimension of the coordinates this dof map provides
    unsigned int geometric_dimension() const;

    /// Return number of facet dofs
    unsigned int num_facet_dofs() const;

    /// Local-to-global mapping of dofs on a cell
    const std::vector<uint>& cell_dofs(uint cell_index) const
    {
      assert(cell_index < dofmap.size());
      return dofmap[cell_index];
    }

    /// Tabulate the local-to-global mapping of dofs on a cell (UFC cell version)
    void tabulate_dofs(uint* dofs, const ufc::cell& ufc_cell, uint cell_index) const;

    /// Tabulate the local-to-global mapping of dofs on a cell (DOLFIN cell version)
    void tabulate_dofs(uint* dofs, const Cell& cell) const;

    /// Tabulate local-local facet dofs
    void tabulate_facet_dofs(uint* dofs, uint local_facet) const;

    /// Tabulate the coordinates of all dofs on a cell (UFC cell version)
    void tabulate_coordinates(double** coordinates, const ufc::cell& ufc_cell) const
    { _ufc_dofmap->tabulate_coordinates(coordinates, ufc_cell); }

    /// Tabulate the coordinates of all dofs on a cell (DOLFIN cell version)
    void tabulate_coordinates(double** coordinates, const Cell& cell) const;

    /// Extract sub dofmap component
    DofMap* extract_sub_dofmap(const std::vector<uint>& component, const Mesh& dolfin_mesh) const;

    /// "Collapse" a sub dofmap
    DofMap* collapse(std::map<uint, uint>& collapsed_map, const Mesh& dolfin_mesh) const;

    /// Return the set of dof indices
    Set<dolfin::uint> dofs(bool sort) const;

    // Renumber dofs
    void renumber(const std::vector<uint>& renumbering_map);

    /// Return informal string representation (pretty-print)
    std::string str(bool verbose) const;

    // Recursively extract UFC sub-dofmap and compute offset
    static ufc::dof_map* extract_sub_dofmap(const ufc::dof_map& ufc_dof_map,
                                            uint& offset,
                                            const std::vector<uint>& component,
                                            const ufc::mesh ufc_mesh,
                                            const Mesh& dolfin_mesh);

  private:

    /// Friends
    friend class DofMapBuilder;

    // Build dofmap from the UFC dofmap
    void build(const Mesh& dolfin_mesh, const UFCMesh& ufc_mesh);

    /// Initialize the UFC dofmap
    static void init_ufc_dofmap(ufc::dof_map& dofmap,
                                const ufc::mesh ufc_mesh,
                                const Mesh& dolfin_mesh);

    // Dof map (dofs for cell dofmap[i])
    std::vector<std::vector<dolfin::uint> > dofmap;

    // Map from UFC dof numbering to renumbered dof (ufc_dof, actual_dof)
    std::map<dolfin::uint, uint> ufc_map_to_dofmap;

    // UFC dof map
    boost::shared_ptr<ufc::dof_map> _ufc_dofmap;

    // UFC dof map offset (this is greater than zero when the dof map is a view,
    // i.e. a sub-dofmap that has not been collapsed)
    unsigned int ufc_offset;

    // True iff running in parallel
    bool _parallel;

  };

}

#endif
