// Copyright (C) 2017 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <dolfinx/graph/Graph.h>
#include <dolfinx/graph/GraphBuilder.h>
#include <dolfinx/mesh/Mesh.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;

namespace dolfinx_wrappers
{
void graph(py::module& m)
{
  // dolfinx::Set
  py::class_<dolfinx::common::Set<int>>(m, "DOLFINIntSet");

  // dolfinx::Graph
  py::class_<dolfinx::graph::Graph>(m, "Graph");

  // dolfinx::GraphBuilder
  py::class_<dolfinx::graph::GraphBuilder>(m, "GraphBuilder")
      .def_static("local_graph",
                  [](const dolfinx::mesh::Mesh& mesh,
                     const std::vector<std::size_t>& coloring) {
                    return dolfinx::graph::GraphBuilder::local_graph(mesh,
                                                                    coloring);
                  })
      .def_static("local_graph", [](const dolfinx::mesh::Mesh& mesh,
                                    std::size_t dim0, std::size_t dim1) {
        return dolfinx::graph::GraphBuilder::local_graph(mesh, dim0, dim1);
      });
}
} // namespace dolfinx_wrappers
