// Copyright (C) 2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-01-30
// Last changed: 2007-01-30

#include <dolfin/dolfin_log.h>
#include <dolfin/Mesh.h>
#include <dolfin/Cell.h>
#include <dolfin/MeshOrdering.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
void MeshOrdering::order(Mesh& mesh)
{
  cout << "Ordering mesh entities..." << endl;

  // Get cell type
  const CellType& cell_type = mesh.type();

  // Iterate over all cells and order the mesh entities locally
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    cell_type.orderEntities(*cell);
  }
}
//-----------------------------------------------------------------------------
