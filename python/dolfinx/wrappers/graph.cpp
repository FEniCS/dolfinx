// Copyright (C) 2017 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "caster_mpi.h"
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/graph/Partitioning.h>
#include <pybind11/eigen.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;

namespace dolfinx_wrappers
{
void graph(py::module& m)
{

  m.def("create_local_adjacency_list",
        &dolfinx::graph::Partitioning::create_local_adjacency_list);
  m.def(
      "create_distributed_adjacency_list",
      [](const MPICommWrapper comm,
         const dolfinx::graph::AdjacencyList<std::int32_t>& list_local,
         const std::vector<std::int64_t>& global_links,
         const std::vector<bool>& exterior_links) {
        return dolfinx::graph::Partitioning::create_distributed_adjacency_list(
            comm.get(), list_local, global_links, exterior_links);
      });
  m.def("distribute",
        [](const MPICommWrapper comm,
           const dolfinx::graph::AdjacencyList<std::int64_t>& list,
           const dolfinx::graph::AdjacencyList<std::int32_t>& destinations) {
          return dolfinx::graph::Partitioning::distribute(comm.get(), list,
                                                          destinations);
        });
  m.def("distribute_data",
        [](const MPICommWrapper comm, const std::vector<std::int64_t>& indices,
           const Eigen::Ref<const Eigen::Array<
               double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& x) {
          return dolfinx::graph::Partitioning::distribute_data<double>(
              comm.get(), indices, x);
        });
  m.def("compute_local_to_global_links",
        &dolfinx::graph::Partitioning::compute_local_to_global_links);
  m.def("compute_local_to_local",
        &dolfinx::graph::Partitioning::compute_local_to_local);

#define ADJACENCYLIST_MACRO(SCALAR, SCALAR_NAME)                               \
  py::class_<dolfinx::graph::AdjacencyList<SCALAR>,                            \
             std::shared_ptr<dolfinx::graph::AdjacencyList<SCALAR>>>(          \
      m, "AdjacencyList_" #SCALAR_NAME, "Adjacency list")                    \
      .def(py::init<std::int32_t>())                                           \
      .def(py::init<const Eigen::Ref<const Eigen::Array<                       \
               SCALAR, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>&>())  \
      .def(                                                                    \
          "links",                                                             \
          [](const dolfinx::graph::AdjacencyList<SCALAR>& self, int i) {       \
            return self.links(i);                                              \
          },                                                                   \
          "Links (edges) of a node",                                           \
          py::return_value_policy::reference_internal)                         \
      .def("array", &dolfinx::graph::AdjacencyList<SCALAR>::array,             \
           py::return_value_policy::reference_internal)                        \
      .def("offsets", &dolfinx::graph::AdjacencyList<SCALAR>::offsets,         \
           "Index to each node in the links array",                            \
           py::return_value_policy::reference_internal)                        \
      .def_property_readonly(                                                  \
          "num_nodes", &dolfinx::graph::AdjacencyList<SCALAR>::num_nodes)      \
      .def("__eq__", &dolfinx::graph::AdjacencyList<SCALAR>::operator==,       \
           py::is_operator())                                                  \
      .def("__repr__", &dolfinx::graph::AdjacencyList<SCALAR>::str)            \
      .def("__len__", &dolfinx::graph::AdjacencyList<SCALAR>::num_nodes);

  ADJACENCYLIST_MACRO(std::int64_t, int64);
  ADJACENCYLIST_MACRO(std::int32_t, int32);
#undef ADJACENCYLIST_MACRO
}
} // namespace dolfinx_wrappers
