// Copyright (C) 2017 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <array>
#include <dolfin/generation/BoxMesh.h>
#include <dolfin/generation/IntervalMesh.h>
#include <dolfin/generation/RectangleMesh.h>
#include <dolfin/generation/UnitDiscMesh.h>
#include <dolfin/generation/UnitTriangleMesh.h>
#include <dolfin/geometry/Point.h>
#include <dolfin/mesh/CellType.h>
#include <iostream>
#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <string>

#include "casters.h"

namespace py = pybind11;

namespace dolfin_wrappers
{

void generation(py::module& m)
{
  // dolfin::generation::IntervalMesh
  py::class_<dolfin::generation::IntervalMesh,
             std::shared_ptr<dolfin::generation::IntervalMesh>>(m,
                                                                "IntervalMesh")
      .def_static("create", [](const MPICommWrapper comm, std::size_t n,
                               std::array<double, 2> p,
                               dolfin::mesh::GhostMode ghost_mode) {
        return dolfin::generation::IntervalMesh::create(comm.get(), n, p, ghost_mode);
      });

  // dolfin::RectangleMesh
  py::class_<dolfin::generation::RectangleMesh,
             std::shared_ptr<dolfin::generation::RectangleMesh>>(
      m, "RectangleMesh")
      .def_static("create",
                  [](const MPICommWrapper comm,
                     std::array<dolfin::geometry::Point, 2> p,
                     std::array<std::size_t, 2> n,
                     dolfin::mesh::CellType::Type cell_type,
                     dolfin::mesh::GhostMode ghost_mode,
                     std::string diagonal) {
                    return dolfin::generation::RectangleMesh::create(
                        comm.get(), p, n, cell_type, ghost_mode, diagonal);
                  },
                  py::arg("comm"), py::arg("p"), py::arg("n"),
                  py::arg("cell_type"), py::arg("ghost_mode"),
                  py::arg("diagonal") = "right");

  // dolfin::UnitTriangleMesh
  py::class_<dolfin::generation::UnitTriangleMesh>(m, "UnitTriangleMesh")
      .def_static("create", &dolfin::generation::UnitTriangleMesh::create);

  // dolfin::UnitDiscMesh
  py::class_<dolfin::generation::UnitDiscMesh>(m, "UnitDiscMesh")
      .def_static("create", [](const MPICommWrapper comm, std::size_t n,
                               dolfin::mesh::GhostMode ghost_mode) {
        return dolfin::generation::UnitDiscMesh::create(comm.get(), n, ghost_mode);
      });

  // dolfin::BoxMesh
  py::class_<dolfin::generation::BoxMesh,
             std::shared_ptr<dolfin::generation::BoxMesh>>(m, "BoxMesh")
      .def_static("create",
                  [](const MPICommWrapper comm,
                     std::array<dolfin::geometry::Point, 2> p,
                     std::array<std::size_t, 3> n,
                     dolfin::mesh::CellType::Type cell_type,
                     const dolfin::mesh::GhostMode ghost_mode) {
                    return dolfin::generation::BoxMesh::create(comm.get(), p, n,
                                                               cell_type,
                                                               ghost_mode);
                  },
                  py::arg("comm"), py::arg("p"), py::arg("n"),
                  py::arg("cell_type"), py::arg("ghost_mode"));
}
}
