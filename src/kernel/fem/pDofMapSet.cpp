// Copyright (C) 2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Magnus Vikstrøm 2008.
//
// First added:  2008-01-11
// Last changed: 2008-01-15

#include <dolfin/dolfin_log.h>
#include <dolfin/pDofMap.h>
#include <dolfin/pDofMapSet.h>
#include <dolfin/Mesh.h>
#include <dolfin/pForm.h>

#include <ufc.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
pDofMapSet::pDofMapSet()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
pDofMapSet::pDofMapSet(const pForm& form, Mesh& mesh, 
    MeshFunction<uint>& partitions)
{
  update(form, mesh, partitions);
}
//-----------------------------------------------------------------------------
pDofMapSet::pDofMapSet(const ufc::form& form, Mesh& mesh,
    MeshFunction<uint>& partitions)
{
  update(form, mesh, partitions);
}
//-----------------------------------------------------------------------------
pDofMapSet::~pDofMapSet()
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
void pDofMapSet::update(const pForm& form, Mesh& mesh,
        MeshFunction<uint>& partitions)
{
  update(form.form(), mesh, partitions);
}
//-----------------------------------------------------------------------------
void pDofMapSet::update(const ufc::form& form, Mesh& mesh,
        MeshFunction<uint>& partitions)
{
  dolfin_debug("Updating set of dof maps...");

  // Resize array of dof maps
  dof_map_set.resize(form.rank());

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
      pDofMap* dolfin_dof_map = new pDofMap(*ufc_dof_map, mesh, partitions);
      dolfin_assert(dolfin_dof_map);

      // Save pair of UFC and DOLFIN dof maps in cache
      std::pair<ufc::dof_map*, pDofMap*> dof_map_pair(ufc_dof_map, dolfin_dof_map);
      dof_map_cache[ufc_dof_map->signature()] = dof_map_pair;
      
      // Set dof map for argument i
      dof_map_set[i] = dolfin_dof_map;
    }
    else
    {
      message(2, "Reusing dof map (already in cache): %s", ufc_dof_map->signature());
      
      // Set dof map for argument i
      dof_map_set[i] = it->second.second;
     
      // Delete UFC dof map (not used)
      delete ufc_dof_map;
    }
  }

  dolfin_debug("Finished updating set of dof maps");
}
//-----------------------------------------------------------------------------
void pDofMapSet::build(pUFC& ufc) const
{
  for (uint i=0; i<dof_map_set.size(); ++i)
    dof_map_set[i]->build(ufc);
}
//-----------------------------------------------------------------------------
dolfin::uint pDofMapSet::size() const
{
  return dof_map_set.size();
}
//-----------------------------------------------------------------------------
pDofMap& pDofMapSet::operator[] (uint i) const
{
  dolfin_assert(i < dof_map_set.size());
  return *dof_map_set[i];
}
//-----------------------------------------------------------------------------

