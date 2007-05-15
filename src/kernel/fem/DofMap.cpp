// Copyright (C) 2007 Anders Logg and Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.

// First added:  2007-03-01
// Last changed: 2007-03-15

#include <dolfin/constants.h>
#include <dolfin/Cell.h>
#include <dolfin/UFCCell.h>
#include <dolfin/DofMap.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
DofMap::DofMap(ufc::dof_map& dof_map, Mesh& mesh) : ufc_dof_map(dof_map), 
               dolfin_mesh(mesh)
{
  // Initialize mesh entities used by dof map
  for (uint d = 0; d <= mesh.topology().dim(); d++)
  {
    if ( ufc_dof_map.needs_mesh_entities(d) )
    {
      mesh.init(d);
      mesh.order();
    }
  }
  
  // Initialize UFC mesh data (must be done after entities are created)
  ufc_mesh.init(mesh);

  // Initialize UFC dof map
  const bool init_cells = ufc_dof_map.init_mesh(ufc_mesh);
  if ( init_cells )
  {
    CellIterator cell(mesh);
    UFCCell ufc_cell(*cell);
    for (; !cell.end(); ++cell)
    {
      ufc_cell.update(*cell);
      ufc_dof_map.init_cell(ufc_mesh, ufc_cell);
    }
    ufc_dof_map.init_cell_finalize();
  }
}
//-----------------------------------------------------------------------------
DofMap::~DofMap()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
