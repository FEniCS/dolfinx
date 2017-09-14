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

#include <memory>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <dolfin/graph/Graph.h>
#include <dolfin/graph/GraphBuilder.h>
#include <dolfin/mesh/Mesh.h>

namespace py = pybind11;

namespace dolfin_wrappers
{
  void graph(py::module& m)
  {
    // dolfin::Set
    py::class_<dolfin::Set<int>>(m, "DOLFINIntSet");

    // dolfin::Graph
    py::class_<dolfin::Graph>(m, "Graph");

    // dolfin::GraphBuilder
    py::class_<dolfin::GraphBuilder>(m, "GraphBuilder")
      .def_static("local_graph", [](const dolfin::Mesh& mesh, const std::vector<std::size_t>& coloring)
                  { return dolfin::GraphBuilder::local_graph(mesh, coloring); })
      .def_static("local_graph", [](const dolfin::Mesh& mesh, std::size_t dim0, std::size_t dim1)
                  { return dolfin::GraphBuilder::local_graph(mesh, dim0, dim1); });
  }
}
