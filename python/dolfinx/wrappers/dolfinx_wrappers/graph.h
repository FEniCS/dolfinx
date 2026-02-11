// Copyright (C) 2017-2025 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "MPICommWrapper.h"
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/graph/ordering.h>
#include <dolfinx/graph/partition.h>
#include <dolfinx/graph/partitioners.h>
#include <dolfinx/graph/utils.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/operators.h>
#include <nanobind/stl/function.h>
#include <span>
#include <string>
#include <vector>

namespace dolfinx_wrappers
{

/// Wrap a C++ graph partitioning function as a Python-ready function
template <typename Functor>
auto create_partitioner_py(Functor&& p_cpp)
{
  return [p_cpp](dolfinx_wrappers::MPICommWrapper comm, int nparts,
                 const dolfinx::graph::AdjacencyList<std::int64_t>& local_graph,
                 bool ghosting)
  { return p_cpp(comm.get(), nparts, local_graph, ghosting); };
}

/// Declare AdjacencyList class with __init__ methods for a given type
/// @param m The nanobind module
/// @param type String representation of the type (e.g., "int32", "int64")
template <typename T, typename U>
void declare_adjacency_list_init(nanobind::module_& m, std::string type)
{
  namespace nb = nanobind;

  std::string pyclass_name = std::string("AdjacencyList_") + type;
  nb::class_<dolfinx::graph::AdjacencyList<T, U>>(m, pyclass_name.c_str(),
                                                  "Adjacency List")
      .def(
          "__init__",
          [](dolfinx::graph::AdjacencyList<T, U>* a,
             nb::ndarray<const T, nb::ndim<1>, nb::c_contig> adj)
          {
            std::vector<T> data(adj.data(), adj.data() + adj.size());
            new (a) dolfinx::graph::AdjacencyList<T, U>(
                dolfinx::graph::regular_adjacency_list<U>(std::move(data), 1));
          },
          nb::arg("adj").noconvert())
      .def(
          "__init__",
          [](dolfinx::graph::AdjacencyList<T, U>* a,
             nb::ndarray<const T, nb::ndim<2>, nb::c_contig> adj)
          {
            std::vector<T> data(adj.data(), adj.data() + adj.size());
            new (a) dolfinx::graph::AdjacencyList<T, U>(
                dolfinx::graph::regular_adjacency_list<U>(std::move(data),
                                                          adj.shape(1)));
          },
          nb::arg("adj").noconvert())
      .def(
          "__init__",
          [](dolfinx::graph::AdjacencyList<T, U>* a,
             nb::ndarray<const T, nb::ndim<1>, nb::c_contig> array,
             nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> displ)
          {
            std::vector<T> data(array.data(), array.data() + array.size());
            std::vector<std::int32_t> offsets(displ.data(),
                                              displ.data() + displ.size());
            new (a) dolfinx::graph::AdjacencyList<T, U>(std::move(data),
                                                        std::move(offsets));
          },
          nb::arg("data").noconvert(), nb::arg("offsets"))
      .def(
          "links",
          [](const dolfinx::graph::AdjacencyList<T, U>& self, int i)
          {
            std::span<const T> link = self.links(i);
            return nb::ndarray<const T, nb::numpy>(link.data(), {link.size()});
          },
          nb::rv_policy::reference_internal, nb::arg("i"),
          "Links (edges) of a node")
      .def_prop_ro(
          "array",
          [](const dolfinx::graph::AdjacencyList<T, U>& self)
          {
            return nb::ndarray<const T, nb::numpy>(self.array().data(),
                                                   {self.array().size()});
          },
          nb::rv_policy::reference_internal)
      .def_prop_ro(
          "offsets",
          [](const dolfinx::graph::AdjacencyList<T, U>& self)
          {
            return nb::ndarray<const std::int32_t, nb::numpy>(
                self.offsets().data(), {self.offsets().size()});
          },
          nb::rv_policy::reference_internal)
      .def_prop_ro("num_nodes", &dolfinx::graph::AdjacencyList<T, U>::num_nodes)
      .def("__eq__", &dolfinx::graph::AdjacencyList<T, U>::operator==,
           nb::is_operator())
      .def("__repr__", &dolfinx::graph::AdjacencyList<T, U>::str)
      .def("__len__", &dolfinx::graph::AdjacencyList<T, U>::num_nodes);
}

/// Declare additional AdjacencyList properties for a given type
/// @param m The nanobind module
/// @param type String representation of the type
template <typename T, typename U>
void declare_adjacency_list(nanobind::module_& m, std::string type)
{
  namespace nb = nanobind;

  std::string pyclass_name = std::string("AdjacencyList_") + type;
  nb::class_<dolfinx::graph::AdjacencyList<T, U>>(m, pyclass_name.c_str(),
                                                  "Adjacency List")
      .def_prop_ro(
          "offsets",
          [](const dolfinx::graph::AdjacencyList<T, U>& self)
          {
            return nb::ndarray<const std::int32_t, nb::numpy>(
                self.offsets().data(), {self.offsets().size()});
          },
          nb::rv_policy::reference_internal)
      .def_prop_ro("num_nodes", &dolfinx::graph::AdjacencyList<T, U>::num_nodes)
      .def("__eq__", &dolfinx::graph::AdjacencyList<T, U>::operator==,
           nb::is_operator())
      .def("__len__", &dolfinx::graph::AdjacencyList<T, U>::num_nodes);
}

} // namespace dolfinx_wrappers
