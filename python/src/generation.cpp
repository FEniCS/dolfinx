// Copyright (C) 2017 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <array>
#include <dolfin/generation/BoxMesh.h>
#include <dolfin/generation/IntervalMesh.h>
#include <dolfin/generation/RectangleMesh.h>
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
  // dolfin::IntervalMesh
  py::class_<dolfin::IntervalMesh, std::shared_ptr<dolfin::IntervalMesh>,
             dolfin::Mesh>(m, "IntervalMesh")
      .def_static("create", [](const MPICommWrapper comm, std::size_t n,
                               std::array<double, 2> p) {
        return dolfin::IntervalMesh::create(comm.get(), n, p);
      });

  // dolfin::RectangleMesh
  py::class_<dolfin::RectangleMesh, std::shared_ptr<dolfin::RectangleMesh>,
             dolfin::Mesh>(m, "RectangleMesh")
      .def_static("create",
                  [](const MPICommWrapper comm, std::array<dolfin::Point, 2> p,
                     std::array<std::size_t, 2> n,
                     dolfin::CellType::Type cell_type, std::string diagonal) {
                    return dolfin::RectangleMesh::create(comm.get(), p, n,
                                                         cell_type, diagonal);
                  },
                  py::arg("comm"), py::arg("p"), py::arg("n"),
                  py::arg("cell_type"), py::arg("diagonal") = "right");

  // dolfin::UnitTriangleMesh
  py::class_<dolfin::UnitTriangleMesh>(m, "UnitTriangleMesh")
      .def_static("create", &dolfin::UnitTriangleMesh::create);

  // dolfin::BoxMesh
  py::class_<dolfin::BoxMesh, std::shared_ptr<dolfin::BoxMesh>, dolfin::Mesh>(
      m, "BoxMesh")
      .def_static(
          "create",
          [](const MPICommWrapper comm, std::array<dolfin::Point, 2> p,
             std::array<std::size_t, 3> n, dolfin::CellType::Type cell_type) {
            return dolfin::BoxMesh::create(comm.get(), p, n, cell_type);
          },
          py::arg("comm"), py::arg("p"), py::arg("n"), py::arg("cell_type"));
}
}
