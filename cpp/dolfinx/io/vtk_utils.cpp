// Copyright (C) 2005-2022 Garth N. Wells and Jørgen S. Dokken
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "vtk_utils.h"
#include "cells.h"
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <span>
#include <tuple>

using namespace dolfinx;

//-----------------------------------------------------------------------------
std::pair<std::vector<std::int64_t>, std::array<std::size_t, 2>>
io::extract_vtk_connectivity(const graph::AdjacencyList<std::int32_t>& dofmap_x,
                             mesh::CellType cell_type)
{
  // Get DOLFINx to VTK permutation
  // FIXME: Use better way to get number of nodes
  // const std::size_t num_nodes = geometry.cmap().dim();
  const std::size_t num_nodes = dofmap_x.links(0).size();
  std::vector vtkmap
      = io::cells::transpose(io::cells::perm_vtk(cell_type, num_nodes));

  // Extract mesh 'nodes'
  const std::size_t num_cells = dofmap_x.num_nodes();

  // Build mesh connectivity

  // Loop over cells
  std::array<std::size_t, 2> shape = {num_cells, num_nodes};
  std::vector<std::int64_t> topology(shape[0] * shape[1]);
  for (std::size_t c = 0; c < num_cells; ++c)
  {
    // For each cell, get the 'nodes' and place in VTK order
    auto dofs_x = dofmap_x.links(c);
    for (std::size_t i = 0; i < dofs_x.size(); ++i)
      topology[c * shape[1] + i] = dofs_x[vtkmap[i]];
  }

  return {std::move(topology), shape};
}
//-----------------------------------------------------------------------------
