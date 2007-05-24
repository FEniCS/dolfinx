// Copyright (C) 2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2007.
//
// First added:  2007-01-17
// Last changed: 2007-05-14

#include <dolfin/dolfin_log.h>
#include <dolfin/Cell.h>
#include <dolfin/Facet.h>
#include <dolfin/DofMap.h>
#include <dolfin/DofMaps.h>
#include <dolfin/Mesh.h>
#include <dolfin/SparsityPattern.h>
#include <dolfin/UFCCell.h>
#include <dolfin/UFCMesh.h>
#include <dolfin/UFC.h>

#include <ufc.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
DofMaps::DofMaps()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
DofMaps::~DofMaps()
{
  // Delete all dof maps in the cache
  for (map_iterator it = dof_map_cache.begin(); it != dof_map_cache.end(); it++)
  {
    // Delete UFC dof map
    delete it->second.first;

    // Delete DOLFIN dof map
    delete it->second.second;
  }
}
//-----------------------------------------------------------------------------
void DofMaps::update(const ufc::form& form, Mesh& mesh)
{
  // Resize array of dof maps
  dof_maps.resize(form.rank());

  // Create dof maps and reuse previously computed dof maps
  for (uint i = 0; i < form.rank(); i++)
  {
    // Create UFC dof map
    ufc::dof_map* ufc_dof_map = form.create_dof_map(i);
    dolfin_assert(ufc_dof_map);
    
    // Check if dof map is in cache
    map_iterator it = dof_map_cache.find(ufc_dof_map->signature());
    if ( it == dof_map_cache.end() )
    {
      message(2, "Creating dof map (not in cache): %s", ufc_dof_map->signature());

      // Create DOLFIN dof map
      DofMap* dolfin_dof_map = new DofMap(*ufc_dof_map, mesh);
      dolfin_assert(dolfin_dof_map);

      // Save pair of UFC and DOLFIN dof maps in cache
      std::pair<ufc::dof_map*, DofMap*> dof_map_pair(ufc_dof_map, dolfin_dof_map);
      dof_map_cache[ufc_dof_map->signature()] = dof_map_pair;
      
      // Set dof map for argument i
      dof_maps[i] = dolfin_dof_map;
    }
    else
    {
      message(2, "Reusing dof map (already in cache): %s", ufc_dof_map->signature());
      
      // Set dof map for argument i
      dof_maps[i] = it->second.second;
     
      // Delete UFC dof map (not used)
      delete ufc_dof_map;
    }
  }
}
//-----------------------------------------------------------------------------
dolfin::uint DofMaps::size() const
{
  return dof_maps.size();
}
//-----------------------------------------------------------------------------
const DofMap& DofMaps::operator[] (uint i) const
{
  dolfin_assert(i < dof_maps.size());
  return *dof_maps[i];
}
//-----------------------------------------------------------------------------
void DofMaps::sparsityPattern(SparsityPattern& sparsity_pattern, Mesh& mesh,
                              UFC& ufc) const
{
  if (size() == 0)
    scalarSparsityPattern(sparsity_pattern);
  else if (size() == 1)
    vectorSparsityPattern(sparsity_pattern);
  else if (size() == 2)
    matrixSparsityPattern(sparsity_pattern, mesh, ufc);
  else
    error("Cannot compute sparsity patterm for size > 2.");
}
//-----------------------------------------------------------------------------
void DofMaps::scalarSparsityPattern(SparsityPattern& sparsity_pattern) const
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void DofMaps::vectorSparsityPattern(SparsityPattern& sparsity_pattern) const
{
  // Get map
  DofMap* dof_map0 = dof_maps[0];

  // Initialise sparsity pattern 
  sparsity_pattern.init(dof_map0->global_dimension());
}
//-----------------------------------------------------------------------------
void DofMaps::matrixSparsityPattern(SparsityPattern& sparsity_pattern, Mesh& mesh, 
                                    UFC& ufc) const
{
  if( size() != 2)
    error("Number of DOF maps in not equal to 2. Do not know how to build sparsity pattern.");
 
  // Initialise sparsity pattern
  sparsity_pattern.init(dof_maps[0]->global_dimension(), dof_maps[1]->global_dimension());

  // Create sparsity pattern for cell integrals
  if (ufc.form.num_interior_facet_integrals() != 0)
  {
    for (CellIterator cell(mesh); !cell.end(); ++cell)
    {
      // Update to current cell
      ufc.update(*cell);
  
      // Tabulate dofs for each dimension
      ufc.dof_maps[0]->tabulate_dofs(ufc.dofs[0], ufc.mesh, ufc.cell);
      ufc.dof_maps[1]->tabulate_dofs(ufc.dofs[1], ufc.mesh, ufc.cell);
  
      // Build sparsity
      uint dim0 = ufc.dof_maps[0]->local_dimension();
      uint dim1 = ufc.dof_maps[1]->local_dimension();
      for (uint i = 0; i < dim0; ++i)
        for (uint j = 0; j < dim1; ++j)
          sparsity_pattern.insert( (ufc.dofs[0])[i], (ufc.dofs[0])[j] );
    }
  }

  // Create sparsity pattern for interior facet integrals
  if(ufc.form.num_interior_facet_integrals() != 0)
  {
    // Compute facets and facet - cell connectivity if not already computed
    mesh.init(mesh.topology().dim() - 1);
    mesh.init(mesh.topology().dim() - 1, mesh.topology().dim());
    mesh.order();
  
    for (FacetIterator facet(mesh); !facet.end(); ++facet)
    {
      // Check if we have an interior facet
      if ( facet->numEntities(mesh.topology().dim()) != 2 )
        continue;

      // Get cells incident with facet
      Cell cell0(mesh, facet->entities(mesh.topology().dim())[0]);
      Cell cell1(mesh, facet->entities(mesh.topology().dim())[1]);
      
      // Update to current pair of cells
      ufc.update(cell0, cell1);
    
      // Tabulate dofs for each dimension on macro element
      for (uint i = 0; i < ufc.form.rank(); i++)
      {
        const uint offset = ufc.local_dimensions[i];
        ufc.dof_maps[i]->tabulate_dofs(ufc.macro_dofs[i], ufc.mesh, ufc.cell0);
        ufc.dof_maps[i]->tabulate_dofs(ufc.macro_dofs[i] + offset, ufc.mesh, ufc.cell1);
      }

      // Build sparsity
      uint dim0 = ufc.macro_local_dimensions[0];
      uint dim1 = ufc.macro_local_dimensions[1];
      for (uint i = 0; i < dim0; ++i)
        for (uint j = 0; j < dim1; ++j)
          sparsity_pattern.insert( (ufc.macro_dofs[0])[i], (ufc.macro_dofs[1])[j] );
    }
  }
}
//-----------------------------------------------------------------------------
