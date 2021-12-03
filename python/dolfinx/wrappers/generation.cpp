// Copyright (C) 2017 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "MPICommWrapper.h"
#include "caster_mpi.h"
#include <array>
#include <dolfinx/fem/CoordinateElement.h>
#include <dolfinx/generation/BoxMesh.h>
#include <dolfinx/generation/IntervalMesh.h>
#include <dolfinx/generation/RectangleMesh.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

namespace py = pybind11;

namespace
{
using PythonCellPartitionFunction
    = std::function<dolfinx::graph::AdjacencyList<std::int32_t>(
        dolfinx_wrappers::MPICommWrapper, int, int,
        const dolfinx::graph::AdjacencyList<std::int64_t>&,
        dolfinx::mesh::GhostMode)>;

using CppCellPartitionFunction
    = std::function<dolfinx::graph::AdjacencyList<std::int32_t>(
        MPI_Comm, int, int, const dolfinx::graph::AdjacencyList<std::int64_t>&,
        dolfinx::mesh::GhostMode)>;

CppCellPartitionFunction
create_cell_partitioner_cpp(const PythonCellPartitionFunction& partitioner)
{
  return [partitioner](MPI_Comm comm, int n, int tdim,
                       const dolfinx::graph::AdjacencyList<std::int64_t>& cells,
                       dolfinx::mesh::GhostMode ghost_mode)
  {
    return partitioner(dolfinx_wrappers::MPICommWrapper(comm), n, tdim, cells,
                       ghost_mode);
  };
}

} // namespace

namespace dolfinx_wrappers
{

void generation(py::module& m)
{
  py::enum_<dolfinx::generation::DiagonalType>(m, "DiagonalType")
      .value("left", dolfinx::generation::DiagonalType::left)
      .value("right", dolfinx::generation::DiagonalType::right)
      .value("crossed", dolfinx::generation::DiagonalType::crossed)
      .value("left_right", dolfinx::generation::DiagonalType::left_right)
      .value("right_left", dolfinx::generation::DiagonalType::right_left);

  m.def(
      "create_interval_mesh",
      [](const MPICommWrapper comm, std::size_t n, std::array<double, 2> p,
         dolfinx::mesh::GhostMode ghost_mode,
         const PythonCellPartitionFunction& partitioner)
      {
        return dolfinx::generation::IntervalMesh::create(
            comm.get(), n, p, ghost_mode,
            create_cell_partitioner_cpp(partitioner));
      },
      py::arg("comm"), py::arg("n"), py::arg("p"), py::arg("ghost_mode"),
      py::arg("partitioner"));

  m.def(
      "create_rectangle_mesh",
      [](const MPICommWrapper comm,
         const std::array<std::array<double, 3>, 2>& p,
         std::array<std::size_t, 2> n, dolfinx::mesh::CellType celltype,
         dolfinx::mesh::GhostMode ghost_mode,
         const PythonCellPartitionFunction& partitioner,
         dolfinx::generation::DiagonalType diagonal)
      {
        return dolfinx::generation::RectangleMesh::create(
            comm.get(), p, n, celltype, ghost_mode,
            create_cell_partitioner_cpp(partitioner), diagonal);
      },
      py::arg("comm"), py::arg("p"), py::arg("n"), py::arg("celltype"),
      py::arg("ghost_mode"), py::arg("partitioner"), py::arg("diagonal"));

  m.def(
      "create_box_mesh",
      [](const MPICommWrapper comm,
         const std::array<std::array<double, 3>, 2>& p,
         std::array<std::size_t, 3> n, dolfinx::mesh::CellType celltype,
         dolfinx::mesh::GhostMode ghost_mode,
         const PythonCellPartitionFunction& partitioner)
      {
        return dolfinx::generation::BoxMesh::create(
            comm.get(), p, n, celltype, ghost_mode,
            create_cell_partitioner_cpp(partitioner));
      },
      py::arg("comm"), py::arg("p"), py::arg("n"), py::arg("celltype"),
      py::arg("ghost_mode"), py::arg("partitioner"));
}
} // namespace dolfinx_wrappers
