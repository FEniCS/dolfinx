// Copyright (C) 2017 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "../common/DolfinXException.h"
#include "caster_mpi.h"
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/graph/ordering.h>
#include <dolfinx/graph/partition.h>
#include <dolfinx/graph/partitioners.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;

namespace
{
/// Wrap a C++ graph partitioning function as a Python-ready function
template <typename Functor>
auto create_partitioner_py(Functor p_cpp)
{
  return [p_cpp](dolfinx_wrappers::MPICommWrapper comm, int nparts,
                 const dolfinx::graph::AdjacencyList<std::int64_t>& local_graph,
                 bool ghosting)
  { return p_cpp(comm.get(), nparts, local_graph, ghosting); };
}
} // namespace

namespace dolfinx_wrappers
{

template <typename T>
void declare_adjacency_list(py::module& m, std::string type)
{
  std::string pyclass_name = std::string("AdjacencyList_") + type;
  py::class_<dolfinx::graph::AdjacencyList<T>,
             std::shared_ptr<dolfinx::graph::AdjacencyList<T>>>(
      m, pyclass_name.c_str(), "Adjacency List")
      .def(py::init(
               [](const py::array_t<T, py::array::c_style>& adj)
               {
                 if (adj.ndim() > 2)
                   throw DolfinXException("Incorrect array dimension.");
                 const std::size_t dim = adj.ndim() < 2 ? 1 : adj.shape(1);
                 std::vector<T> data(adj.data(), adj.data() + adj.size());
                 return dolfinx::graph::regular_adjacency_list(std::move(data),
                                                               dim);
               }),
           py::arg("adj"))
      .def(py::init(
               [](const py::array_t<T, py::array::c_style>& array,
                  const py::array_t<std::int32_t, py::array::c_style>& displ)
               {
                 std::vector<T> data(array.data(), array.data() + array.size());
                 std::vector<std::int32_t> offsets(displ.data(),
                                                   displ.data() + displ.size());
                 return dolfinx::graph::AdjacencyList<T>(std::move(data),
                                                         std::move(offsets));
               }),
           py::arg("data"), py::arg("offsets"))
      .def(
          "links",
          [](const dolfinx::graph::AdjacencyList<T>& self, int i)
          {
            std::span<const T> link = self.links(i);
            return py::array_t<T>(link.size(), link.data(), py::cast(self));
          },
          py::arg("i"), "Links (edges) of a node")
      .def_property_readonly("array",
                             [](const dolfinx::graph::AdjacencyList<T>& self)
                             {
                               return py::array_t<T>(self.array().size(),
                                                     self.array().data(),
                                                     py::cast(self));
                             })
      .def_property_readonly("offsets",
                             [](const dolfinx::graph::AdjacencyList<T>& self)
                             {
                               return py::array_t<std::int32_t>(
                                   self.offsets().size(), self.offsets().data(),
                                   py::cast(self));
                             })
      .def_property_readonly("num_nodes",
                             &dolfinx::graph::AdjacencyList<T>::num_nodes)
      .def("__eq__", &dolfinx::graph::AdjacencyList<T>::operator==,
           py::is_operator())
      .def("__repr__", &dolfinx::graph::AdjacencyList<T>::str)
      .def("__len__", &dolfinx::graph::AdjacencyList<T>::num_nodes);
}

void graph(py::module& m)
{

  declare_adjacency_list<std::int32_t>(m, "int32");
  declare_adjacency_list<std::int64_t>(m, "int64");

  using partition_fn
      = std::function<dolfinx::graph::AdjacencyList<std::int32_t>(
          MPICommWrapper, int,
          const dolfinx::graph::AdjacencyList<std::int64_t>&, bool)>;
  m.def(
      "partitioner",
      []() -> partition_fn
      { return create_partitioner_py(dolfinx::graph::partition_graph); },
      "Default graph partitioner");
#ifdef HAS_PTSCOTCH
  m.def(
      "partitioner_scotch",
      [](double imbalance, int seed) -> partition_fn
      {
        return create_partitioner_py(dolfinx::graph::scotch::partitioner(
            dolfinx::graph::scotch::strategy::none, imbalance, seed));
      },
      py::arg("imbalance") = 0.025, py::arg("seed") = 0,
      "SCOTCH graph partitioner");
#endif
#ifdef HAS_PARMETIS
  m.def(
      "partitioner_parmetis",
      [](double imbalance, std::array<int, 3> options) -> partition_fn
      {
        return create_partitioner_py(
            dolfinx::graph::parmetis::partitioner(imbalance, options));
      },
      py::arg("imbalance") = 1.02,
      py::arg("options") = std ::array<int, 3>({1, 0, 5}),
      "ParMETIS graph partitioner");
#endif
#ifdef HAS_KAHIP
  m.def(
      "partitioner_kahip",
      [](int mode = 1, int seed = 1, double imbalance = 0.03,
         bool suppress_output = true) -> partition_fn
      {
        return create_partitioner_py(dolfinx::graph::kahip::partitioner(
            mode, seed, imbalance, suppress_output));
      },
      py::arg("mode") = 1, py::arg("seed") = 1, py::arg("imbalance") = 0.03,
      py::arg("suppress_output") = true, "KaHIP graph partitioner");
#endif

  m.def("reorder_gps", &dolfinx::graph::reorder_gps, py::arg("graph"));
}
} // namespace dolfinx_wrappers
