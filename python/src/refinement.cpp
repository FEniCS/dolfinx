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

#include <pybind11/pybind11.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/refinement/refine.h>

namespace py = pybind11;

namespace dolfin_wrappers
{
  void refinement(py::module& m)
  {
    // dolfin/refinement free functions
    m.def("refine", (dolfin::Mesh (*)(const dolfin::Mesh&, bool)) &dolfin::refine,
          py::arg("mesh"), py::arg("redistribute")=true);
    m.def("refine", (dolfin::Mesh (*)(const dolfin::Mesh&, const dolfin::MeshFunction<bool>&, bool))
          &dolfin::refine, py::arg("mesh"), py::arg("marker"), py::arg("redistribute")=true);
  }
}
