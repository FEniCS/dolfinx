// Copyright (C) 2017-2026 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "MPICommWrapper.h"
#include <dolfinx/graph/AdjacencyList.h>
#include <functional>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/operators.h>
#include <nanobind/stl/function.h>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace dolfinx_wrappers
{

namespace nb = nanobind;

/// Wrap a C++ graph partitioning function as a Python-ready function.
template <typename Functor>
auto create_partitioner_py(Functor&& p_cpp)
{
  return [p_cpp](dolfinx_wrappers::MPICommWrapper comm, int nparts,
                 const dolfinx::graph::AdjacencyList<std::int64_t>& local_graph,
                 nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig>
                     node_weights,
                 nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig>
                     edge_weights,
                 bool ghosting)
  {
    std::span<const std::int32_t> node_weights_span(node_weights.data(),
                                                    node_weights.size());
    std::span<const std::int32_t> edge_weights_span(edge_weights.data(),
                                                    edge_weights.size());
    return p_cpp(comm.get(), nparts, local_graph, node_weights_span,
                 edge_weights_span, ghosting);
  };
}

/// Wrap a C++ geometric graph partitioner for use from Python. Node
/// coordinates are passed as a 2D array of shape (num_nodes, gdim). The
/// Python signature always requires both the graph and coordinates.
template <typename Functor>
auto create_geom_partitioner_py(Functor&& p_cpp)
{
  return
      [p_cpp](
          dolfinx_wrappers::MPICommWrapper comm, int nparts,
          const dolfinx::graph::AdjacencyList<std::int64_t>& local_graph,
          nanobind::ndarray<const double, nanobind::ndim<2>, nanobind::c_contig>
              x,
          bool ghosting)
  {
    return p_cpp(comm.get(), nparts, std::cref(local_graph),
                 std::span<const double>(x.data(), x.size()), x.shape(1),
                 ghosting);
  };
}

/// Wrap a C++ hybrid graph partitioner for use from Python. As
/// create_geom_partitioner_py, but the wrapped C++ functor's graph
/// argument is required (not optional) and has no separate `gdim`
/// parameter, matching dolfinx::graph::hybrid_partition_fn.
template <typename Functor>
auto create_hybrid_partitioner_py(Functor&& p_cpp)
{
  return
      [p_cpp](
          dolfinx_wrappers::MPICommWrapper comm, int nparts,
          const dolfinx::graph::AdjacencyList<std::int64_t>& local_graph,
          nanobind::ndarray<const double, nanobind::ndim<2>, nanobind::c_contig>
              x,
          bool ghosting)
  {
    return p_cpp(comm.get(), nparts, local_graph,
                 std::span<const double>(x.data(), x.size()), ghosting);
  };
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
            if (displ.size() == 0 or displ.data()[0] != 0)
            {
              throw std::runtime_error(
                  "offsets must be non-empty and start at 0.");
            }

            for (std::size_t i = 1; i < displ.size(); ++i)
            {
              if (displ.data()[i] < displ.data()[i - 1])
                throw std::runtime_error("offsets must be non-decreasing.");
            }

            if (static_cast<std::size_t>(displ.data()[displ.size() - 1])
                != array.size())
            {
              throw std::runtime_error(
                  "Last entry in offsets must equal the length of data.");
            }

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
      .def("__eq__",
           [](const dolfinx::graph::AdjacencyList<T, U>& self, nb::handle other)
           {
             return nb::isinstance<dolfinx::graph::AdjacencyList<T, U>>(other)
                    and self
                            == nb::cast<
                                const dolfinx::graph::AdjacencyList<T, U>&>(
                                other);
           })
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
      .def("__eq__",
           [](const dolfinx::graph::AdjacencyList<T, U>& self, nb::handle other)
           {
             return nb::isinstance<dolfinx::graph::AdjacencyList<T, U>>(other)
                    and self
                            == nb::cast<
                                const dolfinx::graph::AdjacencyList<T, U>&>(
                                other);
           })
      .def("__len__", &dolfinx::graph::AdjacencyList<T, U>::num_nodes);
}

} // namespace dolfinx_wrappers
