// Copyright (C) 2017 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "MPICommWrapper.h"
#include "caster_mpi.h"
#include <array>
#include <dolfinx/generation/BoxMesh.h>
#include <dolfinx/generation/IntervalMesh.h>
#include <dolfinx/generation/RectangleMesh.h>
#include <dolfinx/generation/UnitDiscMesh.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Topology.h>
#include <iostream>
#include <memory>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

namespace py = pybind11;

namespace dolfinx_wrappers
{

void generation(py::module& m)
{
  // dolfinx::generation::IntervalMesh
  py::class_<dolfinx::generation::IntervalMesh,
             std::shared_ptr<dolfinx::generation::IntervalMesh>>(m,
                                                                 "IntervalMesh")
      .def_static(
          "create",
          [](const MPICommWrapper comm, std::size_t n, std::array<double, 2> p,
             dolfinx::mesh::GhostMode ghost_mode) {
            return dolfinx::generation::IntervalMesh::create(comm.get(), n, p,
                                                             ghost_mode);
          },
          py::arg("comm"), py::arg("n"), py::arg("p"), py::arg("ghost_mode"));

  // dolfinx::RectangleMesh
  py::class_<dolfinx::generation::RectangleMesh,
             std::shared_ptr<dolfinx::generation::RectangleMesh>>(
      m, "RectangleMesh")
      .def_static(
          "create",
          [](const MPICommWrapper comm, std::array<Eigen::Vector3d, 2> p,
             std::array<std::size_t, 2> n, dolfinx::mesh::CellType cell_type,
             dolfinx::mesh::GhostMode ghost_mode, std::string diagonal) {
            return dolfinx::generation::RectangleMesh::create(
                comm.get(), p, n, cell_type, ghost_mode, diagonal);
          },
          py::arg("comm"), py::arg("p"), py::arg("n"), py::arg("cell_type"),
          py::arg("ghost_mode"), py::arg("diagonal") = "right");

  // dolfinx::UnitDiscMesh
  py::class_<dolfinx::generation::UnitDiscMesh>(m, "UnitDiscMesh")
      .def_static(
          "create",
          [](const MPICommWrapper comm, std::size_t n,
             dolfinx::mesh::GhostMode ghost_mode) {
            return dolfinx::generation::UnitDiscMesh::create(comm.get(), n,
                                                             ghost_mode);
          },
          py::arg("comm"), py::arg("n"), py::arg("ghost_mode"));

  // dolfinx::BoxMesh
  py::class_<dolfinx::generation::BoxMesh,
             std::shared_ptr<dolfinx::generation::BoxMesh>>(m, "BoxMesh")
      .def_static(
          "create",
          [](const MPICommWrapper comm, std::array<Eigen::Vector3d, 2> p,
             std::array<std::size_t, 3> n, dolfinx::mesh::CellType cell_type,
             const dolfinx::mesh::GhostMode ghost_mode) {
            return dolfinx::generation::BoxMesh::create(comm.get(), p, n,
                                                        cell_type, ghost_mode);
          },
          py::arg("comm"), py::arg("p"), py::arg("n"), py::arg("cell_type"),
          py::arg("ghost_mode"));
}
} // namespace dolfinx_wrappers
