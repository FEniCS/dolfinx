// Copyright (C) 2007-2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-01-30
// Last changed: 2010-10-19

#include <boost/shared_ptr.hpp>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/log/log.h>
#include "Cell.h"
#include "Mesh.h"
#include "ParallelData.h"
#include "MeshOrdering.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void MeshOrdering::order(Mesh& mesh)
{
  info(TRACE, "Ordering mesh.");

  // Special case
  if (mesh.num_cells() == 0)
    return;

  // Get global vertex numbering (important when running in parallel)
  const MeshFunction<unsigned int>* global_vertex_indices = 0;
  if (mesh.parallel_data().have_global_entity_indices(0))
    global_vertex_indices = &(mesh.parallel_data().global_entity_indices(0));

  // Iterate over all cells and order the mesh entities locally
  Progress p("Ordering mesh", mesh.num_cells());
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    cell->order(global_vertex_indices);
    p++;
  }
}
//-----------------------------------------------------------------------------
bool MeshOrdering::ordered(const Mesh& mesh)
{
  // Special case
  if (mesh.num_cells() == 0)
    return true;

  // Get global vertex numbering (important when running in parallel)
  const MeshFunction<unsigned int>* global_vertex_indices = 0;
  if (mesh.parallel_data().have_global_entity_indices(0))
    global_vertex_indices = &(mesh.parallel_data().global_entity_indices(0));

  // Check if all cells are ordered
  Progress p("Checking mesh ordering", mesh.num_cells());
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    if (!cell->ordered(global_vertex_indices))
      return false;
    p++;
  }

  return true;
}
//-----------------------------------------------------------------------------
