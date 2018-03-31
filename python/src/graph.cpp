// Copyright (C) 2017 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <dolfin/graph/Graph.h>
#include <dolfin/graph/GraphBuilder.h>
#include <dolfin/mesh/Mesh.h>
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;

namespace dolfin_wrappers
{
void graph(py::module& m)
{
  // dolfin::Set
  py::class_<dolfin::common::Set<int>>(m, "DOLFINIntSet");

  // dolfin::Graph
  py::class_<dolfin::graph::Graph>(m, "Graph");

  // dolfin::GraphBuilder
  py::class_<dolfin::graph::GraphBuilder>(m, "GraphBuilder")
      .def_static("local_graph",
                  [](const dolfin::mesh::Mesh& mesh,
                     const std::vector<std::size_t>& coloring) {
                    return dolfin::graph::GraphBuilder::local_graph(mesh,
                                                                    coloring);
                  })
      .def_static("local_graph", [](const dolfin::mesh::Mesh& mesh,
                                    std::size_t dim0, std::size_t dim1) {
        return dolfin::graph::GraphBuilder::local_graph(mesh, dim0, dim1);
      });
}
}
