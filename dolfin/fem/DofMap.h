// Copyright (C) 2007-2009 Anders Logg and Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Martin Alnes, 2008
// Modified by Kent-Andre Mardal, 2009
// Modified by Ola Skavhaug, 2009
//
// First added:  2007-03-01
// Last changed: 2009-08-14

#ifndef __DOF_MAP_H
#define __DOF_MAP_H

#include <map>
#include <memory>
#include <vector>
#include <boost/shared_ptr.hpp>
#include <dolfin/common/types.h>
#include <dolfin/common/Variable.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshFunction.h>
#include "UFC.h"
#include "UFCCell.h"
#include "UFCMesh.h"

namespace dolfin
{

  class UFC;

  /// This class handles the mapping of degrees of freedom.
  /// It wraps a ufc::dof_map on a specific mesh and provides
  /// optional precomputation and reordering of dofs.

  class DofMap : public Variable
  {
  public:

    /// Create dof map on mesh
    DofMap(boost::shared_ptr<ufc::dof_map> ufc_dof_map, 
           boost::shared_ptr<Mesh> mesh);

    /// Create dof map on mesh (const mesh version)
    DofMap(boost::shared_ptr<ufc::dof_map> ufc_dof_map, 
           boost::shared_ptr<const Mesh> mesh);

  private:

    /// Create dof map on mesh with a std::vector dof map
    DofMap(std::auto_ptr<std::vector<int> > map, 
           boost::shared_ptr<ufc::dof_map> ufc_dof_map, 
           boost::shared_ptr<const Mesh> mesh);

  public:

    /// Destructor
    ~DofMap();

    /// Return a string identifying the dof map
    std::string signature() const
    {
      if (!map.get())
        return ufc_dof_map->signature();
      else
      {
        error("DofMap has been re-ordered. Cannot return signature string.");
        return ufc_dof_map->signature();
      }
    }

    /// Return true iff mesh entities of topological dimension d are needed
    bool needs_mesh_entities(unsigned int d) const
    { return ufc_dof_map->needs_mesh_entities(d); }

    /// Return the dimension of the global finite element function space
    unsigned int global_dimension() const
    {
      assert(_global_dimension > 0);
      return _global_dimension;
    }

    /// Return the dimension of the local finite element function space on a cell
    unsigned int local_dimension(const ufc::cell& cell) const
    { return ufc_dof_map->local_dimension(cell); }

    /// Return the maximum dimension of the local finite element function space
    unsigned int max_local_dimension() const
    { return ufc_dof_map->max_local_dimension(); }

    /// Return number of facet dofs
    unsigned int num_facet_dofs() const
    { return ufc_dof_map->num_facet_dofs(); }

    /// Tabulate the local-to-global mapping of dofs on a cell
    void tabulate_dofs(uint* dofs, const ufc::cell& ufc_cell, uint cell_index) const;

    /// Tabulate local-local facet dofs
    void tabulate_facet_dofs(uint* dofs, uint local_facet) const;

    /// Tabulate the coordinates of all dofs on a cell
    void tabulate_coordinates(double** coordinates, const ufc::cell& ufc_cell) const
    { ufc_dof_map->tabulate_coordinates(coordinates, ufc_cell); }

    /// Extract sub dofmap and offset for component
    DofMap* extract_sub_dofmap(const std::vector<uint>& component, uint& offset) const;

    /// Test whether dof map has been renumbered
    bool renumbered() const
    {
      if (map.get())
        return true;
      else
        return false;
    }

    /// Return informal string representation (pretty-print)
    std::string str(bool verbose) const;

  private:

    /// Friends
    friend class DofMapBuilder;

    /// Initialise UFC dof map
    void init_ufc();

    // Recursively extract sub dofmap
    ufc::dof_map* extract_sub_dofmap(const ufc::dof_map& dof_map,
                                     uint& offset,
                                     const std::vector<uint>& component) const;

    // FIXME: Should this be a std::vector<std::vector<int> >, 
    //        e.g. a std::vector for each cell? 
    // FIXME: Document layout of map
    // Precomputed dof map 
    std::auto_ptr<std::vector<int> > map;

    // Global dimension
    uint _global_dimension;

    // Map from UFC dofs to renumbered dof 
    std::map<dolfin::uint, int> ufc_to_map;

    // UFC dof map
    boost::shared_ptr<ufc::dof_map> ufc_dof_map;

    // UFC mesh
    UFCMesh ufc_mesh;

    // UFC dof map offset into parent's vector of coefficients
    uint _ufc_offset;

    // Mesh we live in
    boost::shared_ptr<const Mesh> dolfin_mesh;

    // True iff running in parallel
    bool parallel;

  };

}

#endif
