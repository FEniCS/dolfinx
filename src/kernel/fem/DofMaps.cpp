// Copyright (C) 2007 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2007-01-17
// Last changed: 2007-03-01

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
  std::map<const std::string, std::pair<ufc::dof_map*, DofMap*> >::iterator it;
  for (it = dof_map_cache.begin(); it != dof_map_cache.end(); it++)
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
  const uint num_arguments = form.rank() + form.num_coefficients();
  dof_maps.resize(num_arguments);

  // Create dof maps and reuse previously computed dof maps
  for (uint i = 0; i < num_arguments; i++)
  {
    // Create UFC dof map
    ufc::dof_map* ufc_dof_map = form.create_dof_map(i);

    // Check if dof map is in cache
    if ( dof_map_cache.find(ufc_dof_map->signature()) == dof_map_cache.end() )
    {
      cout << "Creating dof map (not in cache): " << ufc_dof_map->signature() << endl;

      // Create DOLFIN dof map
      DofMap* dolfin_dof_map = new DofMap(*ufc_dof_map, mesh);

      // Save pair of UFC and DOLFIN dof maps in cache
      std::pair<ufc::dof_map*, DofMap*> dof_map_pair(ufc_dof_map, dolfin_dof_map);
      dof_map_cache[ufc_dof_map->signature()] = dof_map_pair;
      
      // Set dof map for argument i
      dof_maps[i] = dolfin_dof_map;
    }
    else
    {
      cout << "Reusing dof map (already in cache): " << ufc_dof_map->signature() << endl;
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
