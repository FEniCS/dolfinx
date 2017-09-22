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

#include <iostream>
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <dolfin/geometry/Point.h>
#include <dolfin/generation/BoxMesh.h>
#include <dolfin/generation/UnitTriangleMesh.h>
#include <dolfin/generation/UnitCubeMesh.h>
#include <dolfin/generation/UnitDiscMesh.h>
#include <dolfin/generation/SphericalShellMesh.h>
#include <dolfin/generation/UnitSquareMesh.h>
#include <dolfin/generation/UnitIntervalMesh.h>
#include <dolfin/generation/UnitQuadMesh.h>
#include <dolfin/generation/UnitHexMesh.h>
#include <dolfin/generation/IntervalMesh.h>

#include "casters.h"

namespace py = pybind11;

namespace dolfin_wrappers
{

  void generation(py::module& m)
  {
    // dolfin::IntervalMesh
    py::class_<dolfin::IntervalMesh, std::shared_ptr<dolfin::IntervalMesh>, dolfin::Mesh>(m, "IntervalMesh")
      .def(py::init<std::size_t, double, double>())
      .def(py::init<MPI_Comm, std::size_t, double, double>());

    // dolfin::UnitIntervalMesh
    py::class_<dolfin::UnitIntervalMesh, std::shared_ptr<dolfin::UnitIntervalMesh>,
               dolfin::IntervalMesh, dolfin::Mesh>(m, "UnitIntervalMesh")
      .def(py::init<std::size_t>())
      .def(py::init<MPI_Comm, std::size_t>())
      .def_static("create", [](std::size_t n){ return dolfin::UnitIntervalMesh::create(n); });

    // dolfin::RectangleMesh
    py::class_<dolfin::RectangleMesh, std::shared_ptr<dolfin::RectangleMesh>, dolfin::Mesh>(m, "RectangleMesh")
      .def(py::init<dolfin::Point, dolfin::Point, std::size_t, std::size_t, std::string>(),
           py::arg("p0"), py::arg("p1"), py::arg("nx"), py::arg("ny"), py::arg("diagonal")="right")
      .def(py::init<MPI_Comm, dolfin::Point, dolfin::Point, std::size_t, std::size_t, std::string>(),
           py::arg("comm"), py::arg("p0"), py::arg("p1"), py::arg("nx"), py::arg("ny"),
           py::arg("diagonal")="right");

    // dolfin::UnitSquareMesh
    py::class_<dolfin::UnitSquareMesh, std::shared_ptr<dolfin::UnitSquareMesh>, dolfin::Mesh>(m, "UnitSquareMesh")
      .def(py::init<std::size_t, std::size_t>())
      .def(py::init<MPI_Comm, std::size_t, std::size_t>())
      .def(py::init<std::size_t, std::size_t, std::string>())
      .def(py::init<MPI_Comm, std::size_t, std::size_t, std::string>());

    // dolfin::UnitCubeMesh
    py::class_<dolfin::UnitCubeMesh, std::shared_ptr<dolfin::UnitCubeMesh>, dolfin::Mesh>(m, "UnitCubeMesh")
      .def(py::init<std::size_t, std::size_t, std::size_t>())
      .def(py::init<MPI_Comm, std::size_t, std::size_t, std::size_t>());

    // dolfin::UnitDiscMesh
    py::class_<dolfin::UnitDiscMesh>(m, "UnitDiscMesh")
      .def_static("create", &dolfin::UnitDiscMesh::create);

    // dolfin::SphericalShellMesh
    py::class_<dolfin::SphericalShellMesh>(m, "SphericalShellMesh")
      .def_static("create", &dolfin::SphericalShellMesh::create);

    // dolfin::UnitTriangleMesh
    py::class_<dolfin::UnitTriangleMesh>(m, "UnitTriangleMesh")
      .def_static("create", &dolfin::UnitTriangleMesh::create);

    // dolfin::UnitQuadMesh
    py::class_<dolfin::UnitQuadMesh>(m, "UnitQuadMesh")
      .def_static("create", [](std::size_t nx, std::size_t ny)
                  { return dolfin::UnitQuadMesh::create(nx, ny); })
      .def_static("create", [](MPI_Comm comm, std::size_t nx, std::size_t ny)
                  { return dolfin::UnitQuadMesh::create(comm, nx, ny); });

    // dolfin::UnitHexMesh
    py::class_<dolfin::UnitHexMesh>(m, "UnitHexMesh")
      .def_static("create", [](std::size_t nx, std::size_t ny, std::size_t nz)
                  { return dolfin::UnitHexMesh::create(nx, ny, nz); })
      .def_static("create", [](MPI_Comm comm, std::size_t nx, std::size_t ny, std::size_t nz)
                  { return dolfin::UnitHexMesh::create(comm, nx, ny, nz); });

    // dolfin::BoxMesh
    py::class_<dolfin::BoxMesh, std::shared_ptr<dolfin::BoxMesh>, dolfin::Mesh>(m, "BoxMesh")
      .def(py::init<const dolfin::Point&, const dolfin::Point&, std::size_t, std::size_t, std::size_t>())
      .def(py::init<MPI_Comm, const dolfin::Point&, const dolfin::Point&, std::size_t, std::size_t, std::size_t>());
  }
}
