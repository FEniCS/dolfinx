// Copyright (C) 2007-2012 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "MeshOrdering.h"
#include "Cell.h"
#include "Mesh.h"
#include "MeshIterator.h"
#include <dolfin/log/log.h>
#include <memory>
#include <vector>

using namespace dolfin;

//-----------------------------------------------------------------------------
void MeshOrdering::order(Mesh& mesh)
{
  log(TRACE, "Ordering mesh.");

  // Special case
  if (mesh.num_cells() == 0)
    return;

  // Get global vertex numbering
  dolfin_assert(mesh.topology().have_global_indices(0));
  const auto& local_to_global_vertex_indices
      = mesh.topology().global_indices(0);

  // Skip ordering for dimension 0
  if (mesh.topology().dim() == 0)
    return;

  // Iterate over all cells and order the mesh entities locally
  for (auto& cell : MeshRange<Cell>(mesh, MeshRangeType::ALL))
    cell.order(local_to_global_vertex_indices);
}
//-----------------------------------------------------------------------------
bool MeshOrdering::ordered(const Mesh& mesh)
{
  // Special case
  if (mesh.num_cells() == 0)
    return true;

  // Get global vertex numbering
  dolfin_assert(mesh.topology().have_global_indices(0));
  const auto& local_to_global_vertex_indices
      = mesh.topology().global_indices(0);

  // Check if all cells are ordered
  for (auto& cell : MeshRange<Cell>(mesh, MeshRangeType::ALL))
  {
    if (!cell.ordered(local_to_global_vertex_indices))
      return false;
  }

  return true;
}
//-----------------------------------------------------------------------------
