// Copyright (C) 2021 Jørgen S. Dokken
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "adios2_utils.h"
#include "cells.h"
#include "utils.h"
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/mesh/Mesh.h>

using namespace dolfinx;
using namespace dolfinx::io;

//-----------------------------------------------------------------------------
xt::xtensor<std::uint64_t, 2>
adios2_utils::extract_connectivity(std::shared_ptr<const mesh::Mesh> mesh)
{
  // Get DOLFINx to VTK permutation
  // FIXME: Use better way to get number of nodes
  const graph::AdjacencyList<std::int32_t>& x_dofmap
      = mesh->geometry().dofmap();
  const std::uint32_t num_nodes = x_dofmap.num_links(0);
  std::vector map = dolfinx::io::cells::transpose(
      dolfinx::io::cells::perm_vtk(mesh->topology().cell_type(), num_nodes));
  // TODO: Remove when when paraview issue 19433 is resolved
  // (https://gitlab.kitware.com/paraview/paraview/issues/19433)
  if (mesh->topology().cell_type() == dolfinx::mesh::CellType::hexahedron
      and num_nodes == 27)
  {
    map = {0,  9, 12, 3,  1, 10, 13, 4,  18, 15, 21, 6,  19, 16,
           22, 7, 2,  11, 5, 14, 8,  17, 20, 23, 24, 25, 26};
  }
  // Extract mesh 'nodes'
  const int tdim = mesh->topology().dim();
  const std::uint32_t num_cells
      = mesh->topology().index_map(tdim)->size_local();

  // Write mesh connectivity
  xt::xtensor<std::uint64_t, 2> topology({num_cells, num_nodes});
  for (size_t c = 0; c < num_cells; ++c)
  {
    auto x_dofs = x_dofmap.links(c);
    auto top_row = xt::row(topology, c);
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
      top_row[i] = x_dofs[map[i]];
  }

  return topology;
}
//-----------------------------------------------------------------------------
