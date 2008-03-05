// Copyright (C) 2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-01-30
// Last changed: 2007-05-16

#include <dolfin/log/log.h>
#include "Mesh.h"
#include "Cell.h"
#include "MeshOrdering.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void MeshOrdering::order(Mesh& mesh)
{
  message("Ordering mesh entities...");

  // Get cell type
  const CellType& cell_type = mesh.type();

  // Iterate over all cells and order the mesh entities locally
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    cell_type.orderEntities(*cell);
  }
}
//-----------------------------------------------------------------------------
