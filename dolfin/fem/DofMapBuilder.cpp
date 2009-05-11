// Copyright (C) 2008 Anders Logg and Ola Skavhaug.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-08-12
// Last changed: 2009-05-11

#include <algorithm>
#include <dolfin/log/log.h>
#include <dolfin/mesh/Edge.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshData.h>
#include <dolfin/mesh/MeshPartitioning.h>
#include "UFC.h"
#include "DofMap.h"
#include "DofMapBuilder.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void DofMapBuilder::build(DofMap& dof_map, UFC& ufc, Mesh& mesh)
{
  info("Building parallel dof map");

  // Check that dof map has not been built
  if (dof_map.dof_map)
    error("Local-to-global mapping has already been computed.");

  // FIXME: Perhaps the entities need to be numbered from elsewhere
  // FIXME: so the mesh can be const here

  // Number mesh entities globally
  for (uint d = 1; d <= mesh.topology().dim(); ++d)
  {
    if (dof_map.needs_mesh_entities(d))
      MeshPartitioning::number_entities(mesh, d);
  }

  // Allocate dof map
  delete [] dof_map.dof_map;
  const uint n = dof_map.max_local_dimension();
  dof_map.dof_map = new int[n*mesh.num_cells()];

  // Fill dof map 
  //uint offset = 0;
  UFCCell ufc_cell(mesh);
  for (CellIterator c(mesh); !c.end(); ++c)
  {
    ufc_cell.update(*c);
  }
}
//-----------------------------------------------------------------------------
