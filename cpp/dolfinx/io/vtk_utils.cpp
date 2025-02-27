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
io::extract_vtk_connectivity(
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        const std::int32_t,
        MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
        dofmap_x,
    mesh::CellType cell_type)
{
  // Get DOLFINx to VTK permutation
  const std::size_t num_nodes = dofmap_x.extent(1);
  std::vector vtkmap
      = io::cells::transpose(io::cells::perm_vtk(cell_type, num_nodes));

  // Extract mesh 'nodes'
  const std::size_t num_cells = dofmap_x.extent(0);

  // Build mesh connectivity

  // Loop over cells
  std::array<std::size_t, 2> shape = {num_cells, num_nodes};
  std::vector<std::int64_t> topology(shape[0] * shape[1]);
  for (std::size_t c = 0; c < num_cells; ++c)
  {
    // For each cell, get the 'nodes' and place in VTK order
    for (std::size_t i = 0; i < num_nodes; ++i)
      topology[c * shape[1] + i] = dofmap_x(c, vtkmap[i]);
  }

  return {std::move(topology), shape};
}
//-----------------------------------------------------------------------------
std::int8_t io::get_vtk_cell_type(mesh::CellType cell, int dim)
{
  if (cell == mesh::CellType::prism and dim == 2)
    throw std::runtime_error("More work needed for prism cell");

  // Get cell type
  mesh::CellType cell_type = mesh::cell_entity_type(cell, dim, 0);

  // Determine VTK cell type (arbitrary Lagrange elements)
  // https://vtk.org/doc/nightly/html/vtkCellType_8h_source.html
  switch (cell_type)
  {
  case mesh::CellType::point:
    return 1;
  case mesh::CellType::interval:
    return 68;
  case mesh::CellType::triangle:
    return 69;
  case mesh::CellType::quadrilateral:
    return 70;
  case mesh::CellType::tetrahedron:
    return 71;
  case mesh::CellType::hexahedron:
    return 72;
  case mesh::CellType::pyramid:
    return 14;
  case mesh::CellType::prism:
    return 73;
  default:
    throw std::runtime_error("Unknown cell type");
  }
}
//-----------------------------------------------------------------------------
