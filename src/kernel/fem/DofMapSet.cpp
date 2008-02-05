// Copyright (C) 2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2007.
//
// First added:  2007-01-17
// Last changed: 2007-05-24

#include <dolfin/dolfin_log.h>
#include <dolfin/DofMap.h>
#include <dolfin/DofMapSet.h>
#include <dolfin/Mesh.h>
#include <dolfin/Form.h>

#include <ufc.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
DofMapSet::DofMapSet()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
DofMapSet::DofMapSet(const Form& form, Mesh& mesh) : _parallel(false)
{
  update(form.form(), mesh);
}
//-----------------------------------------------------------------------------
DofMapSet::DofMapSet(const Form& form, Mesh& mesh, 
                     MeshFunction<uint>& partitions) : _parallel(true)
{
  update(form, mesh, partitions);
}
//-----------------------------------------------------------------------------
DofMapSet::DofMapSet(const ufc::form& form, Mesh& mesh) : _parallel(false)
{
  update(form, mesh);
}
//-----------------------------------------------------------------------------
DofMapSet::DofMapSet(const ufc::form& form, Mesh& mesh, 
                     MeshFunction<uint>& partitions) : _parallel(true)
{
  update(form, mesh, partitions);
}
//-----------------------------------------------------------------------------
DofMapSet::~DofMapSet()
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
void DofMapSet::update(const Form& form, Mesh& mesh)
{
  update(form.form(), mesh);
}
//-----------------------------------------------------------------------------
void DofMapSet::update(const Form& form, Mesh& mesh, MeshFunction<uint>& partitions)
{
  update(form.form(), mesh, partitions);
}
//-----------------------------------------------------------------------------
void DofMapSet::update(const ufc::form& form, Mesh& mesh)
{
  dolfin_debug("Updating set of dof maps...");

  const uint num_arguments = form.rank() +
    form.num_coefficients();

  // Resize array of dof maps
  dof_map_set.resize(num_arguments);

  // Create dof maps and reuse previously computed dof maps
  for (uint i = 0; i < num_arguments; i++)
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
void DofMapSet::update(const ufc::form& form, Mesh& mesh,
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
      DofMap* dolfin_dof_map = new DofMap(*ufc_dof_map, mesh, partitions);
      dolfin_assert(dolfin_dof_map);

      // Save pair of UFC and DOLFIN dof maps in cache
      std::pair<ufc::dof_map*, DofMap*> dof_map_pair(ufc_dof_map, dolfin_dof_map);
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
void DofMapSet::build(UFC& ufc) const
{
  for (uint i=0; i<dof_map_set.size(); ++i)
    dof_map_set[i]->build(ufc);
}
//-----------------------------------------------------------------------------
dolfin::uint DofMapSet::size() const
{
  return dof_map_set.size();
}
//-----------------------------------------------------------------------------
DofMap& DofMapSet::operator[] (uint i) const
{
  dolfin_assert(i < dof_map_set.size());
  return *dof_map_set[i];
}
//-----------------------------------------------------------------------------

