// Copyright (C) 2017 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "array.h"
#include "caster_mpi.h"
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/graph/ordering.h>
#include <dolfinx/graph/partition.h>
#include <dolfinx/graph/partitioners.h>
#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/operators.h>
#include <nanobind/stl/vector.h>
#include <vector>

namespace nb = nanobind;

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

using T = std::int64_t;
template <typename T>
void declare_adjacency_list(nb::module_& m, std::string type)
{
  std::string pyclass_name = std::string("AdjacencyList_") + type;
  nb::class_<dolfinx::graph::AdjacencyList<T>>(m, pyclass_name.c_str(),
                                               "Adjacency List")
      .def(
          "__init__",
          [](dolfinx::graph::AdjacencyList<T>* a,
             const nb::ndarray<T, nb::numpy>& adj)
          {
            if (adj.ndim() > 2)
              throw std::runtime_error("Incorrect array dimension.");
            const std::size_t dim = adj.ndim() < 2 ? 1 : adj.shape(1);
            std::size_t size = adj.shape(0) * dim;
            std::vector<T> data(adj.data(), adj.data() + size);
            new (a) dolfinx::graph::AdjacencyList<T>(
                dolfinx::graph::regular_adjacency_list(std::move(data), dim));
          },
          nb::arg("adj"))
      .def(
          "__init__",
          [](dolfinx::graph::AdjacencyList<T>* a,
             const nb::ndarray<T, nb::numpy>& array,
             const nb::ndarray<std::int32_t, nb::numpy>& displ)
          {
            assert(array.ndim() == 1);
            std::vector<T> data(array.data(), array.data() + array.shape(0));
            assert(displ.ndim() == 1);
            std::vector<std::int32_t> offsets(displ.data(),
                                              displ.data() + displ.shape(0));
            new (a) dolfinx::graph::AdjacencyList<T>(std::move(data),
                                                     std::move(offsets));
          },
          nb::arg("data"), nb::arg("offsets"))
      .def("ret_numpy",
           []()
           {
             size_t shape[2] = {2, 4};
             const float data[8] = {1, 2, 3, 4, 5, 6, 7, 8};
             return nb::ndarray<nb::numpy, const float, nb::shape<2, nb::any>>(
                 data, /* ndim = */ 2, shape);
           })
      .def(
          "links",
          [](const dolfinx::graph::AdjacencyList<T>& self, int i)
          {
            std::span<const T> link = self.links(i);
            const T* ptr = link.data();
            std::size_t shape[1] = {link.size()};
            return nb::ndarray<nb::numpy, const T, nb::shape<nb::any>>(ptr, 1,
                                                                       shape);
          },
          nb::arg("i"), "Links (edges) of a node")
      .def_prop_ro("array",
                   [](const dolfinx::graph::AdjacencyList<T>& self)
                   {
                     const std::vector<T>& array = self.array();
                     const std::size_t size = array.size();
                     return nb::ndarray<nb::numpy, const T, nb::shape<nb::any>>(
                         array.data(), 1, &size);
                   })
      .def_prop_ro("offsets",
                   [](const dolfinx::graph::AdjacencyList<T>& self)
                   {
                     const std::vector<std::int32_t>& offsets = self.offsets();
                     const std::size_t size = offsets.size();
                     return nb::ndarray<nb::numpy, const std::int32_t>(
                         offsets.data(), 1, &size);
                   })
      .def_prop_ro("num_nodes", &dolfinx::graph::AdjacencyList<T>::num_nodes)
      .def("__eq__", &dolfinx::graph::AdjacencyList<T>::operator==,
           nb::is_operator())
      .def("__repr__", &dolfinx::graph::AdjacencyList<T>::str)
      .def("__len__", &dolfinx::graph::AdjacencyList<T>::num_nodes);
}

void graph(nb::module_& m)
{

  //  declare_adjacency_list<std::int32_t>(m, "int32");
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
      nb::arg("imbalance") = 0.025, nb::arg("seed") = 0,
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
      nb::arg("imbalance") = 1.02,
      nb::arg("options") = std ::array<int, 3>({1, 0, 5}),
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
      nb::arg("mode") = 1, nb::arg("seed") = 1, nb::arg("imbalance") = 0.03,
      nb::arg("suppress_output") = true, "KaHIP graph partitioner");
#endif

  m.def("reorder_gps", &dolfinx::graph::reorder_gps, nb::arg("graph"));
}
} // namespace dolfinx_wrappers
