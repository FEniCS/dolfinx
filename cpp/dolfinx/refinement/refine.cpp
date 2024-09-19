// Copyright (C) 2024 Paul T. Kühner
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "refine.h"

namespace dolfinx::refinement
{

graph::AdjacencyList<std::int32_t> maintain_coarse_partitioner(
    MPI_Comm comm, int, const std::vector<mesh::CellType>& cell_types,
    const std::vector<std::span<const std::int64_t>>& cells)
{
  int mpi_rank = MPI::rank(comm);
  int num_cell_vertices = mesh::num_cell_vertices(cell_types.front());
  std::int32_t num_cells = cells.front().size() / num_cell_vertices;
  std::vector<std::int32_t> destinations(num_cells, mpi_rank);
  std::vector<std::int32_t> dest_offsets(num_cells + 1);
  std::iota(dest_offsets.begin(), dest_offsets.end(), 0);
  return graph::AdjacencyList(std::move(destinations), std::move(dest_offsets));
}

} // namespace dolfinx::refinement
