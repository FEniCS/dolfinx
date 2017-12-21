// Copyright (C) 2017 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.

#include <array>
#include <iostream>
#include <memory>
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <dolfin/geometry/Point.h>
#include <dolfin/mesh/CellType.h>
#include <dolfin/generation/BoxMesh.h>
#include <dolfin/generation/IntervalMesh.h>
#include <dolfin/generation/RectangleMesh.h>
#include <dolfin/generation/UnitTriangleMesh.h>
#include <dolfin/generation/UnitDiscMesh.h>
#include <dolfin/generation/SphericalShellMesh.h>

#include "casters.h"

namespace py = pybind11;

namespace dolfin_wrappers
{

  void generation(py::module& m)
  {
    // dolfin::IntervalMesh
    py::class_<dolfin::IntervalMesh, std::shared_ptr<dolfin::IntervalMesh>, dolfin::Mesh>(m, "IntervalMesh")
      .def_static("create", [](const MPICommWrapper comm, std::size_t n, std::array<double, 2> p)
                  { return dolfin::IntervalMesh::create(comm.get(), n, p); });

    // dolfin::RectangleMesh
    py::class_<dolfin::RectangleMesh, std::shared_ptr<dolfin::RectangleMesh>, dolfin::Mesh>(m, "RectangleMesh")
      .def_static("create", [](const MPICommWrapper comm, std::array<dolfin::Point, 2> p,
                               std::array<std::size_t, 2> n, dolfin::CellType::Type cell_type,
                               std::string diagonal)
                  { return dolfin::RectangleMesh::create(comm.get(), p, n, cell_type, diagonal); },
                  py::arg("comm"), py::arg("p"), py::arg("n"), py::arg("cell_type"),
                  py::arg("diagonal")="right");

    // dolfin::UnitDiscMesh
    py::class_<dolfin::UnitDiscMesh>(m, "UnitDiscMesh")
      .def_static("create", [](const MPICommWrapper comm, std::size_t n, std::size_t degree, std::size_t gdim)
                  { return dolfin::UnitDiscMesh::create(comm.get(), n, degree, gdim); });

    // dolfin::SphericalShellMesh
    py::class_<dolfin::SphericalShellMesh>(m, "SphericalShellMesh")
      .def_static("create", [](const MPICommWrapper comm, std::size_t degree)
                  { return dolfin::SphericalShellMesh::create(comm.get(), degree); });

    // dolfin::UnitTriangleMesh
    py::class_<dolfin::UnitTriangleMesh>(m, "UnitTriangleMesh")
      .def_static("create", &dolfin::UnitTriangleMesh::create);

    // dolfin::BoxMesh
    py::class_<dolfin::BoxMesh, std::shared_ptr<dolfin::BoxMesh>, dolfin::Mesh>(m, "BoxMesh")
      .def_static("create", [](const MPICommWrapper comm, std::array<dolfin::Point, 2> p,
                              std::array<std::size_t, 3> n, dolfin::CellType::Type cell_type)
                           { return dolfin::BoxMesh::create(comm.get(), p, n, cell_type); },
                  py::arg("comm"), py::arg("p"), py::arg("n"), py::arg("cell_type"));
  }
}
