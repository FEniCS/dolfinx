// Copyright (C) 2007-2008 Anders Logg and Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.

// Modified by Martin Alnes, 2008

// First added:  2007-03-01
// Last changed: 2008-10-09

#ifndef __DOF_MAP_H
#define __DOF_MAP_H

#include <map>
#include <tr1/memory>
#include <ufc.h>
#include <dolfin/common/types.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshFunction.h>
#include "UFCCell.h"
#include "UFCMesh.h"

namespace dolfin
{

  class SubSystem;
  class UFC;

  /// This class handles the mapping of degrees of freedom.
  /// It wraps a ufc::dof_map on a specific mesh and provides
  /// optional precomputation and reordering of dofs.

  class DofMap
  {
  public:

    /// Create dof map on mesh
    DofMap(ufc::dof_map& dof_map, Mesh& mesh);

    /// Create dof map on mesh (may share ufc::dof_map)
    DofMap(std::tr1::shared_ptr<ufc::dof_map> dof_map, Mesh& mesh);

    /// Create dof map on mesh (parallel)
    DofMap(ufc::dof_map& dof_map, Mesh& mesh, MeshFunction<uint>& partitions);

    /// Create dof map on mesh (may share ufc::dof_map) (parallel)
    DofMap(std::tr1::shared_ptr<ufc::dof_map> dof_map, Mesh& mesh, MeshFunction<uint>& partitions);

    /// Create dof map on mesh
    DofMap(const std::string signature, Mesh& mesh);

    /// Create dof map on mesh (parallel)
    DofMap(const std::string signature, Mesh& mesh, MeshFunction<uint>& partitions);

    /// Destructor
    ~DofMap();

    /// Return a string identifying the dof map
    const char* signature() const
    { 
      if (!dof_map)
        return ufc_dof_map->signature(); 
      else
      {
        error("DofMap has been re-ordered. Cannot return signature string.");
        return ufc_dof_map->signature(); 
      }  
    }
    
    /// Return the dimension of the global finite element function space
    unsigned int global_dimension() const
    { return ufc_dof_map->global_dimension(); }

    /// Return the dimension of the local finite element function space
    unsigned int local_dimension() const
    { return ufc_dof_map->local_dimension(); }

    /// Return the dimension of the local finite element function space
    unsigned int macro_local_dimension() const
    { return ufc_dof_map->local_dimension(); }

    /// Return number of facet dofs
    unsigned int num_facet_dofs() const
    { return ufc_dof_map->num_facet_dofs(); }

    /// Tabulate the local-to-global mapping of dofs on a cell
    void tabulate_dofs(uint* dofs, ufc::cell& ufc_cell, uint cell_index) const;

    /// Tabulate local-local facet dofs
    void tabulate_facet_dofs(uint* dofs, uint local_facet) const
    { ufc_dof_map->tabulate_facet_dofs(dofs, local_facet); }

    // FIXME: Can this function eventually be removed?
    /// Tabulate the local-to-global mapping of dofs on a ufc cell
    void tabulate_dofs(uint* dofs, const ufc::cell& cell) const 
    { ufc_dof_map->tabulate_dofs(dofs, ufc_mesh, cell); }

    void tabulate_coordinates(double** coordinates, const ufc::cell& ufc_cell) const
    { ufc_dof_map->tabulate_coordinates(coordinates, ufc_cell); }

    /// Extract sub dof map
    DofMap* extractDofMap(const Array<uint>& sub_system, uint& offset) const;

    /// Return mesh associated with map
    Mesh& mesh() const
    { return dolfin_mesh; }

    /// Build parallel dof map
    void build(UFC& ufc);
    
    /// Return renumbering (used for testing)
    std::map<uint, uint> getMap() const;
    
    /// Display mapping
    void disp() const;
   
  private:

    /// Friends
    friend class DofMapBuilder;

    /// Initialise DofMap
    void init();
    
    /// Extract sub DofMap
    ufc::dof_map* extractDofMap(const ufc::dof_map& dof_map, uint& offset, const Array<uint>& sub_system) const;

    // Precomputed dof map
    uint* dof_map;

    // UFC dof map
    std::tr1::shared_ptr<ufc::dof_map> ufc_dof_map;

    // UFC mesh
    UFCMesh ufc_mesh;

    // DOLFIN mesh
    Mesh& dolfin_mesh;

    // Number of cells in the mesh
    uint num_cells;

    // Partitions
    MeshFunction<uint>* partitions;

    // Provide easy access to map for testing
    std::map<uint, uint> map;

  };

}

#endif
