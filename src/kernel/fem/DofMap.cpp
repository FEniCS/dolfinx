// Copyright (C) 2007 Anders Logg and Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.

// First added:  2007-03-01
// Last changed: 2007-03-15

#include <dolfin/constants.h>
#include <dolfin/Cell.h>
#include <dolfin/UFCCell.h>
#include <dolfin/DofMap.h>
#include <dolfin/SubSystem.h>
#include <dolfin/Array.h>
#include <dolfin/ElementLibrary.h>
#include <dolfin/UFC.h>
#include <dolfin/MPI.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
DofMap::DofMap(ufc::dof_map& dof_map, Mesh& mesh) : dof_map(0), 
               ufc_dof_map(&dof_map), ufc_dof_map_local(false), 
               dolfin_mesh(mesh), ufc_map(true), num_cells(mesh.numCells()), 
               partitions(0)
{
  init();
}
//-----------------------------------------------------------------------------
DofMap::DofMap(ufc::dof_map& dof_map, Mesh& mesh, MeshFunction<uint>& partitions)
             : dof_map(0), ufc_dof_map(&dof_map), ufc_dof_map_local(false), 
               dolfin_mesh(mesh), ufc_map(true), num_cells(mesh.numCells()), 
               partitions(&partitions)
{
  init();
}
//-----------------------------------------------------------------------------
DofMap::DofMap(const std::string signature, Mesh& mesh) : dof_map(0), 
              ufc_dof_map(0), ufc_dof_map_local(false), dolfin_mesh(mesh), 
              ufc_map(true), num_cells(mesh.numCells()), partitions(0)
{
  // Create ufc dof map from signature
  ufc_dof_map = ElementLibrary::create_dof_map(signature);
  if (!ufc_dof_map)
    error("Unable to find dof map in library: \"%s\".",signature.c_str());

  // Take resposibility for ufc dof map
  ufc_dof_map_local = ufc_dof_map;

  init();
}
//-----------------------------------------------------------------------------
DofMap::DofMap(const std::string signature, Mesh& mesh, 
               MeshFunction<uint>& partitions) : dof_map(0), ufc_dof_map(0), 
               ufc_dof_map_local(false), dolfin_mesh(mesh), ufc_map(true),
               num_cells(mesh.numCells()), partitions(&partitions)
{
  // Create ufc dof map from signature
  ufc_dof_map = ElementLibrary::create_dof_map(signature);
  if (!ufc_dof_map)
    error("Unable to find dof map in library: \"%s\".",signature.c_str());

  // Take resposibility for ufc dof map
  ufc_dof_map_local = ufc_dof_map;

  init();
}
//-----------------------------------------------------------------------------
DofMap::~DofMap()
{
  if (dof_map)
  {
    for (uint i = 0; i < num_cells; ++i)
      delete [] dof_map[i];
    delete [] dof_map;
  }

  if (ufc_dof_map_local)
    delete ufc_dof_map_local;
}
//-----------------------------------------------------------------------------
DofMap* DofMap::extractDofMap(const Array<uint>& sub_system, uint& offset) const
{
  // Check that dof map has not be re-ordered
  if (!ufc_map)
    error("Dof map has been re-ordered. Don't yet know how to extract sub dof maps.");

  // Reset offset
  offset = 0;

  // Recursively extract sub dof map
  ufc::dof_map* sub_dof_map = extractDofMap(*ufc_dof_map, offset, sub_system);
  message(2, "Extracted dof map for sub system: %s", sub_dof_map->signature());
  message(2, "Offset for sub system: %d", offset);

  if(partitions)
    return new DofMap(*sub_dof_map, dolfin_mesh, *partitions);
  else
    return new DofMap(*sub_dof_map, dolfin_mesh);
}
//-----------------------------------------------------------------------------
ufc::dof_map* DofMap::extractDofMap(const ufc::dof_map& dof_map, uint& offset, const Array<uint>& sub_system) const
{
  // Check if there are any sub systems
  if (dof_map.num_sub_dof_maps() == 0)
    error("Unable to extract sub system (there are no sub systems).");

  // Check that a sub system has been specified
  if (sub_system.size() == 0)
    error("Unable to extract sub system (no sub system specified).");
  
  // Check the number of available sub systems
  if (sub_system[0] >= dof_map.num_sub_dof_maps())
    error("Unable to extract sub system %d (only %d sub systems defined).",
                  sub_system[0], dof_map.num_sub_dof_maps());

  // Add to offset if necessary
  for (uint i = 0; i < sub_system[0]; i++)
  {
    ufc::dof_map* ufc_dof_map = dof_map.create_sub_dof_map(i);
    // FIXME: Can we avoid creating a DofMap here just for getting the global dimension?
    if(partitions)
      DofMap dof_map_test(*ufc_dof_map, dolfin_mesh, *partitions);
    else
      DofMap dof_map_test(*ufc_dof_map, dolfin_mesh);
    offset += ufc_dof_map->global_dimension();
    delete ufc_dof_map;
  }
  
  // Create sub system
  ufc::dof_map* sub_dof_map = dof_map.create_sub_dof_map(sub_system[0]);
  
  // Return sub system if sub sub system should not be extracted
  if (sub_system.size() == 1)
    return sub_dof_map;

  // Otherwise, recursively extract the sub sub system
  Array<uint> sub_sub_system;
  for (uint i = 1; i < sub_system.size(); i++)
    sub_sub_system.push_back(sub_system[i]);
  ufc::dof_map* sub_sub_dof_map = extractDofMap(*sub_dof_map, offset, sub_sub_system);
  delete sub_dof_map;

  return sub_sub_dof_map;
}
//-----------------------------------------------------------------------------
void DofMap::init()
{
  dolfin_debug("Initializing dof map...");

  // Order vertices, so entities will be created correctly according to convention
  dolfin_mesh.order();

  // Initialize mesh entities used by dof map
  for (uint d = 0; d <= dolfin_mesh.topology().dim(); d++)
    if ( ufc_dof_map->needs_mesh_entities(d) )
      dolfin_mesh.init(d);
  
  // Initialize UFC mesh data (must be done after entities are created)
  ufc_mesh.init(dolfin_mesh);

  // Initialize UFC dof map
  const bool init_cells = ufc_dof_map->init_mesh(ufc_mesh);
  if ( init_cells )
  {
    CellIterator cell(dolfin_mesh);
    UFCCell ufc_cell(*cell);
    for (; !cell.end(); ++cell)
    {
      ufc_cell.update(*cell);
      ufc_dof_map->init_cell(ufc_mesh, ufc_cell);
    }
    ufc_dof_map->init_cell_finalize();
  }

  // Initialise ufc cell 
  CellIterator cell(dolfin_mesh);
  ufc_cell.init(*cell);

  dolfin_debug("Dof map initialized");
}
//-----------------------------------------------------------------------------
void DofMap::build(UFC& ufc)
{
  dolfin_debug("pDofMap::build()");
  dof_map = new uint*[dolfin_mesh.numCells()];
  
  // for all processes
  uint current_dof = 0;
  for (uint p = 0; p < MPI::numProcesses(); ++p)
  {
    // for all cells
    for (CellIterator c(dolfin_mesh); !c.end(); ++c)
    {
      // if cell in partition belonging to process p
      if ((*partitions)(*c) != p)
        continue;
 
      dof_map[c->index()] = new uint[local_dimension()];
      //dolfin_debug2("cpu %d building cell %d", MPI::processNumber(), c->index());
      ufc.update(*c);
      ufc_dof_map->tabulate_dofs(ufc.dofs[0], ufc.mesh, ufc.cell);

      for (uint i=0; i < ufc_dof_map->local_dimension(); ++i)
      {
        const uint dof = ufc.dofs[0][i];
        //dolfin_debug3("ufc.dofs[%d][%d] = %d", 0, MPI::processNumber(), ufc.dofs[0][i]);

        std::map<uint, uint>::iterator it = map.find(dof);
        if (it != map.end())
        {
          //dolfin_debug2("cpu %d dof %d already computed", MPI::processNumber(), it->second);
          dof_map[c->index()][i] = it->second;
        }
        else
        {
          dof_map[c->index()][i] = current_dof;
          map[dof] = current_dof++;
        }
      }
    }  
  }
}
//-----------------------------------------------------------------------------
std::map<dolfin::uint, dolfin::uint> DofMap::getMap() const
{
  return map;
}
//-----------------------------------------------------------------------------







