// Copyright (C) 2008 Anders Logg and Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Magnus Vikstrøm, 2008
//
// First added:  2008-01-11
// Last changed: 2008-02-04

#ifndef __P_DOF_MAP_H
#define __P_DOF_MAP_H

#include <ufc.h>
#include <dolfin/Mesh.h>
#include <dolfin/UFCCell.h>
#include <dolfin/UFCMesh.h>
#include <dolfin/constants.h>
#include <dolfin/MPI.h>
#include <dolfin/MeshFunction.h>
#include <dolfin/pUFC.h>

namespace dolfin
{
  class SubSytem;

  /// This class handles the mapping of degrees of freedom.
  /// It wraps a ufc::dof_map on a specific mesh and provides
  /// optional precomputation and reordering of dofs.

  class pDofMap
  {
  public:

    /// Create dof map on mesh
    pDofMap(ufc::dof_map& dof_map, Mesh& mesh, MeshFunction<uint>& partitions);

    /// Create dof map on mesh
    pDofMap(const std::string signature, Mesh& mesh, 
        MeshFunction<uint>& partitions);

    /// Destructor
    ~pDofMap();

    /// Return a string identifying the dof map
    const char* signature() const
      { 
        if(ufc_map)
          return ufc_dof_map->signature(); 
        else
        {
          error("pDofMap has been re-ordered. Cannot return signature string.");
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
    void tabulate_dofs(uint* dofs, Cell& cell) 
    {
      if (dof_map)
      {
        for (uint i = 0; i < local_dimension(); i++)
          dofs[i] = dof_map[cell.index()][i];
      }
      else
      {
        ufc_cell.update(cell);
        ufc_dof_map->tabulate_dofs(dofs, ufc_mesh, ufc_cell);
      }
    }

    /// Tabulate local-local facet dofs
    void tabulate_facet_dofs(uint* dofs, uint local_facet) const
    { ufc_dof_map->tabulate_facet_dofs(dofs, local_facet); }

    // FIXME: Can this function eventually be removed?
    /// Tabulate the local-to-global mapping of dofs on a ufc cell
    void tabulate_dofs(uint* dofs, const ufc::cell& cell) const 
      { ufc_dof_map->tabulate_dofs(dofs, ufc_mesh, cell); }

    void tabulate_coordinates(real** coordinates, const ufc::cell& ufc_cell) const
      { ufc_dof_map->tabulate_coordinates(coordinates, ufc_cell); }

    /// Extract sub dof map
    pDofMap* extractDofMap(const Array<uint>& sub_system, uint& offset) const;

    /// Return mesh associated with map
    Mesh& mesh() const
      { return dolfin_mesh; }

    /// Build parallel dof map
    void build(pUFC& ufc);

    /// Return renumbering
    std::map<const uint, uint> getMap() const
      { return map; }

  private:

    /// Initialise pDofMap
    void init();

    /// Extract sub pDofMap
    ufc::dof_map* extractDofMap(const ufc::dof_map& dof_map, uint& offset, const Array<uint>& sub_system) const;

    // Parallel dof map
    uint** dof_map;

    // UFC dof map
    ufc::dof_map* ufc_dof_map;

    // Local UFC dof map
    ufc::dof_map* ufc_dof_map_local;

    // UFC mesh
    UFCMesh ufc_mesh;

    // DOLFIN mesh
    Mesh& dolfin_mesh;

    // UFC cell
    UFCCell ufc_cell;

    bool ufc_map;

    // Number of cells in the mesh
    uint num_cells;

    // Partitions
    MeshFunction<uint>* partitions;

    // Provide easy access to map for testing
    std::map<const uint, uint> map;

  };

}

#endif
