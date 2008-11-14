// Copyright (C) 2007-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-01-30
// Last changed: 2008-11-13

#include <dolfin/log/log.h>
#include "Mesh.h"
#include "Cell.h"
#include "MeshOrdering.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void MeshOrdering::order(Mesh& mesh)
{
  // Iterate over all cells and order the mesh entities locally
  Progress p("Ordering mesh", mesh.numCells());
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    cell->order();
    p++;
  }
}
//-----------------------------------------------------------------------------
bool MeshOrdering::ordered(const Mesh& mesh)
{
  // Check if all cells are ordered
  Progress p("Checking mesh ordering", mesh.numCells());
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    if (!cell->ordered())
      return false;
    p++;
  }

  return true;
}
//-----------------------------------------------------------------------------
