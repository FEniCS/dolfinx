// Copyright (C) 2017 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/graph/Graph.h>
#include <dolfinx/graph/GraphBuilder.h>
#include <dolfinx/mesh/Mesh.h>
#include <pybind11/eigen.h>
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

  // dolfinx::graph::AdjacencyList class
  py::class_<dolfinx::graph::AdjacencyList<std::int64_t>,
             std::shared_ptr<dolfinx::graph::AdjacencyList<std::int64_t>>>(
      m, "AdjacencyList64", "Adjacency list")
      .def(py::init<std::int32_t>())
      .def(py::init<const Eigen::Ref<
               const Eigen::Array<std::int64_t, Eigen::Dynamic, Eigen::Dynamic,
                                  Eigen::RowMajor>>&>())
      .def(
          "links",
          [](const dolfinx::graph::AdjacencyList<std::int64_t>& self, int i) {
            return self.links(i);
          },
          "Links (edges) of a node",
          py::return_value_policy::reference_internal)
      .def("array", &dolfinx::graph::AdjacencyList<std::int64_t>::array)
      .def_property_readonly(
          "num_nodes", &dolfinx::graph::AdjacencyList<std::int64_t>::num_nodes);

  // dolfinx::graph::AdjacencyList class
  py::class_<dolfinx::graph::AdjacencyList<std::int32_t>,
             std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>>>(
      m, "AdjacencyList", "Adjacency list")
      .def(py::init<std::int32_t>())
      // .def(py::init<const Eigen::Ref<const Eigen::Array<std::int32_t,
      // Eigen::Dynamic, Eigen::Dynamic,
      //                                     Eigen::RowMajor>>&>())
      .def(
          "links",
          [](const dolfinx::graph::AdjacencyList<std::int32_t>& self, int i) {
            return self.links(i);
          },
          "Links (edges) of a node",
          py::return_value_policy::reference_internal)
      .def("array", &dolfinx::graph::AdjacencyList<std::int32_t>::array,
           "All edges", py::return_value_policy::reference_internal)
      .def("offsets", &dolfinx::graph::AdjacencyList<std::int32_t>::offsets,
           "Index to each node in the links array",
           py::return_value_policy::reference_internal)
      .def_property_readonly(
          "num_nodes", &dolfinx::graph::AdjacencyList<std::int32_t>::num_nodes);

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
