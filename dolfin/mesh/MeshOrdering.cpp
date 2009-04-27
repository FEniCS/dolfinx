// Copyright (C) 2007-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-01-30
// Last changed: 2008-11-14

#include <dolfin/log/log.h>
#include "Mesh.h"
#include "Cell.h"
#include "MeshOrdering.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void MeshOrdering::order(Mesh& mesh)
{
  // Special case
  if (mesh.num_cells() == 0)
    return;

  // Iterate over all cells and order the mesh entities locally
  Progress p("Ordering mesh", mesh.num_cells());
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    cell->order();
    p++;
  }
}
//-----------------------------------------------------------------------------
bool MeshOrdering::ordered(const Mesh& mesh)
{
  // Special case
  if (mesh.num_cells() == 0)
    return true;

  // Check if all cells are ordered
  Progress p("Checking mesh ordering", mesh.num_cells());
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    if (!cell->ordered())
      return false;
    p++;
  }

  return true;
}
//-----------------------------------------------------------------------------
