// Copyright (C) 2017 Chris Richardson and Garth N. Wells
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

#include <memory>
#include <pybind11/pybind11.h>

#include <dolfin/ale/ALE.h>
#include <dolfin/ale/MeshDisplacement.h>
#include <dolfin/common/Variable.h>
#include <dolfin/function/Expression.h>
#include <dolfin/function/GenericFunction.h>
#include <dolfin/mesh/BoundaryMesh.h>
#include <dolfin/mesh/Mesh.h>

namespace py = pybind11;

namespace dolfin_wrappers
{
  // Interface for dolfin/ale
  void ale(py::module& m)
  {
    // dolfin::MeshDisplacement
    py::class_<dolfin::MeshDisplacement, std::shared_ptr<dolfin::MeshDisplacement>,
               dolfin::Expression>(m, "MeshDisplacement")
      .def(py::init<std::shared_ptr<const dolfin::Mesh>>());

    // ALE static functions
    py::class_<dolfin::ALE>
      (m, "ALE")
      .def_static("move", [](std::shared_ptr<dolfin::Mesh> mesh, const dolfin::BoundaryMesh bmesh)
                  { return dolfin::ALE::move(mesh, bmesh); })
      .def_static("move", [](std::shared_ptr<dolfin::Mesh> mesh0, const dolfin::Mesh& mesh1)
                  { return dolfin::ALE::move(mesh0, mesh1); })
      .def_static("move", [](dolfin::Mesh& mesh, const dolfin::GenericFunction& disp)
                  { dolfin::ALE::move(mesh, disp); })
      .def_static("move", [](dolfin::Mesh& mesh, const py::object disp)
                  {
                    auto _disp = disp.attr("_cpp_object").cast<const dolfin::GenericFunction*>();
                    dolfin::ALE::move(mesh, *_disp);
                  });
  }
}
