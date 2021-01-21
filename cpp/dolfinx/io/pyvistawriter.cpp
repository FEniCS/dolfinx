// Copyright (C) 2020 Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "pyvistawriter.h"
#include <dolfinx/io/VTKWriter.h>
#include <dolfinx/io/cells.h>
#include <dolfinx/mesh/cell_types.h>
#include <dolfinx/mesh/utils.h>
using namespace dolfinx;

std::pair<Eigen::Array<std::int32_t, Eigen::Dynamic, 1>,
          Eigen::Array<std::int8_t, Eigen::Dynamic, 1>>
io::create_pyvista_topology(const mesh::Mesh& mesh, int dim,
                            std::vector<std::int32_t>& entities)
{
  const std::int32_t num_cells = entities.size();
  Eigen::Array<std::int32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      geometry_entities
      = mesh::entities_to_geometry(mesh, dim, entities, false);
  std::int8_t cell_type = io::cells::get_vtk_cell_type(mesh, dim);
  std::vector<std::int8_t> cell_types(num_cells);
  std::fill(cell_types.begin(), cell_types.end(), cell_type);

  // Get dolfin->vtk permutation
  const std::int32_t num_nodes = geometry_entities.cols();
  const mesh::CellType e_type
      = mesh::cell_entity_type(mesh.topology().cell_type(), dim);
  std::vector map_vtk
      = io::cells::transpose(io::cells::perm_vtk(e_type, num_nodes));

  // Convert [[node_01, ..., node_0M],...[node_N1, ..., node_NM]] into
  // [M, node_01, ..., node_0M, M, node_11, ..., M, node_N1, node_NM]
  std::vector<std::int32_t> flattened_cells((num_nodes + 1) * num_cells);
  for (std::int32_t i = 0; i < num_cells; ++i)
  {
    flattened_cells[i * (num_nodes + 1)] = num_nodes;
    auto cell_nodes = geometry_entities.row(i);
    for (std::int32_t j = 0; j < num_nodes; ++j)
      flattened_cells[i * (num_nodes + 1) + 1 + j] = cell_nodes[map_vtk[j]];
  }
  return {Eigen::Map<Eigen::Array<std::int32_t, Eigen::Dynamic, 1>>(
              flattened_cells.data(), flattened_cells.size()),
          Eigen::Map<Eigen::Array<std::int8_t, Eigen::Dynamic, 1>>(
              cell_types.data(), cell_types.size())};
}
