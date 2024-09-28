// Copyright (C) 2017-2024 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "MPICommWrapper.h"
#include <dolfinx/mesh/cell_types.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

namespace dolfinx_wrappers::part::impl
{
/// Wrap a Python graph partitioning function as a C++ function
template <typename Functor>
auto create_partitioner_cpp(Functor p)
{
  return [p](MPI_Comm comm, int nparts,
             const dolfinx::graph::AdjacencyList<std::int64_t>& local_graph,
             bool ghosting)
  {
    return p(dolfinx_wrappers::MPICommWrapper(comm), nparts, local_graph,
             ghosting);
  };
}

/// Wrap a C++ cell partitioning function as a Python function
template <typename Functor>
auto create_cell_partitioner_py(Functor p)
{
  return [p](dolfinx_wrappers::MPICommWrapper comm, int n,
             const std::vector<dolfinx::mesh::CellType>& cell_types,
             std::vector<nb::ndarray<const std::int64_t, nb::numpy>> cells_nb)
  {
    std::vector<std::span<const std::int64_t>> cells;
    std::ranges::transform(
        cells_nb, std::back_inserter(cells), [](auto c)
        { return std::span<const std::int64_t>(c.data(), c.size()); });
    return p(comm.get(), n, cell_types, cells);
  };
}

using PythonCellPartitionFunction
    = std::function<dolfinx::graph::AdjacencyList<std::int32_t>(
        dolfinx_wrappers::MPICommWrapper, int,
        const std::vector<dolfinx::mesh::CellType>&,
        std::vector<nb::ndarray<const std::int64_t, nb::numpy>>)>;

using CppCellPartitionFunction
    = std::function<dolfinx::graph::AdjacencyList<std::int32_t>(
        MPI_Comm, int, const std::vector<dolfinx::mesh::CellType>& q,
        const std::vector<std::span<const std::int64_t>>&)>;

/// Wrap a Python cell graph partitioning function as a C++ function
inline CppCellPartitionFunction
create_cell_partitioner_cpp(const PythonCellPartitionFunction& p)
{
  if (p)
  {
    return [p](MPI_Comm comm, int n,
               const std::vector<dolfinx::mesh::CellType>& cell_types,
               const std::vector<std::span<const std::int64_t>>& cells)
    {
      std::vector<nb::ndarray<const std::int64_t, nb::numpy>> cells_nb;
      std::ranges::transform(
          cells, std::back_inserter(cells_nb),
          [](auto c)
          {
            return nb::ndarray<const std::int64_t, nb::numpy>(
                c.data(), {c.size()}, nb::handle());
          });

      return p(dolfinx_wrappers::MPICommWrapper(comm), n, cell_types, cells_nb);
    };
  }
  else
    return nullptr;
}
} // namespace dolfinx_wrappers::part::impl
