// Copyright (C) 2007 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2007-01-17
// Last changed: 2007-01-17

#include <dolfin/dolfin_log.h>
#include <dolfin/DofMap.h>
#include <dolfin/DofMaps.h>

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
  for (std::map<const std::string, DofMap*>::iterator it = dof_map_cache.begin();
       it != dof_map_cache.end(); it++)
  {
    delete it->second;
  }
}
//-----------------------------------------------------------------------------
void DofMaps::update(const ufc::form& form, Mesh& mesh)
{
  // Resize array of dof maps
  const uint num_dof_maps = form.rank() + form.num_coefficients();
  dof_maps.resize(num_dof_maps);

  // Create dof maps and reuse previously computed maps
  for (uint i = 0; i < num_dof_maps; i++)
  {
    // Create UFC dof map
    ufc::dof_map* ufc_dof_map = form.create_dof_map(i);

    // Create DOLFIN dof map if not in cache
    if ( dof_map_cache.find(ufc_dof_map->signature()) == dof_map_cache.end() )
    {
      DofMap* dolfin_dof_map = new DofMap(mesh, *ufc_dof_map);
      dof_map_cache[ufc_dof_map->signature()] = dolfin_dof_map;
      dof_maps[i] = dolfin_dof_map;
      cout << "Creating dof map (not in cache): " << ufc_dof_map->signature() << endl;
    }
    else
    {
      cout << "Reusing dof map (already in cache): " << ufc_dof_map->signature() << endl;
    }

    // Delete UFC dof map
    delete ufc_dof_map;
  }
}
//-----------------------------------------------------------------------------
const DofMap& DofMaps::operator[] (uint i) const
{
  dolfin_assert(i < dof_maps.size());
  return *dof_maps[i];
}
//-----------------------------------------------------------------------------
