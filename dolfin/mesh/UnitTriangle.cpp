// Copyright (C) 2010 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2010-10-19
// Last changed: 2010-10-19

#include <dolfin/common/MPI.h>
#include "MeshPartitioning.h"
#include "MeshEditor.h"
#include "UnitTriangle.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
UnitTriangle::UnitTriangle() : Mesh()
{
  // Receive mesh according to parallel policy
  if (MPI::is_receiver()) { MeshPartitioning::partition(*this); return; }

  // Open mesh for editing
  MeshEditor editor;
  editor.open(*this, CellType::triangle, 2, 2);

  // Create vertices
  editor.init_vertices(3);
  editor.add_vertex(0, 0, 0);
  editor.add_vertex(1, 1, 0);
  editor.add_vertex(2, 0, 1);

  // Create cells
  editor.init_cells(1);
  editor.add_cell(0, 0, 1, 2);

  // Close mesh editor
  editor.close();

  // Broadcast mesh according to parallel policy
  if (MPI::is_broadcaster()) { MeshPartitioning::partition(*this); return; }
}
//-----------------------------------------------------------------------------
