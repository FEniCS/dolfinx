// Copyright (C) 2017-2026 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "MPICommWrapper.h"
#include "array.h"
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/graph/ordering.h>
#include <dolfinx/graph/partition.h>
#include <functional>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/operators.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace dolfinx_wrappers
{

namespace nb = nanobind;

/// Shape of a plain Python graph partitioning function, as passed to
/// create_mesh or wrapped by partitioner_wrap_py_to_cpp below.
using PythonPartitionFunction
    = std::function<dolfinx::graph::AdjacencyList<std::int32_t>(
        MPICommWrapper, int, const dolfinx::graph::AdjacencyList<std::int64_t>&,
        std::optional<nb::ndarray<const std::int32_t, nb::numpy>>,
        std::optional<nb::ndarray<const std::int32_t, nb::numpy>>, bool)>;

/// Shape of a plain Python geometric graph partitioning function, as
/// wrapped by partitioner_wrap_py_to_cpp / create_geometric_cell_partitioner
/// below.
using PythonGeoPartitionFunction
    = std::function<nb::ndarray<const int, nb::ndim<1>, nb::numpy>(
        MPICommWrapper, int, nb::ndarray<const double, nb::ndim<2>, nb::numpy>,
        std::optional<nb::ndarray<const std::int32_t, nb::numpy>>)>;

/// Shape of a plain Python hybrid graph partitioning function, as
/// wrapped by partitioner_wrap_py_to_cpp / create_hybrid_cell_partitioner
/// below.
using PythonHybridPartitionFunction
    = std::function<dolfinx::graph::AdjacencyList<std::int32_t>(
        MPICommWrapper, int, const dolfinx::graph::AdjacencyList<std::int64_t>&,
        nb::ndarray<const double, nb::ndim<2>, nb::numpy>,
        std::optional<nb::ndarray<const std::int32_t, nb::numpy>>,
        std::optional<nb::ndarray<const std::int32_t, nb::numpy>>, bool)>;

/// Wrap a Python graph partitioning function as a C++ function.
/// std::nullopt is passed through as Python None rather than a
/// zero-length array, so a custom partitioner can distinguish "no
/// weights given" from "empty weights array" -- matching
/// graph::partition_fn's own std::optional weights. Defined in
/// graph.cpp.
dolfinx::graph::partition_fn
partitioner_wrap_py_to_cpp(const PythonPartitionFunction& p);

/// Wrap a Python geometric graph partitioning function as a C++
/// function. `gdim` is not received directly by the Python callable --
/// it is recovered from `x`'s second dimension when reshaping the flat
/// `x` span for the call. A std::nullopt weight is passed through as
/// Python None, matching graph::geom_partition_fn's own std::optional
/// weights. Defined in graph.cpp.
dolfinx::graph::geom_partition_fn
partitioner_wrap_py_to_cpp(const PythonGeoPartitionFunction& p);

/// Wrap a Python hybrid graph partitioning function as a C++ function.
/// `gdim` is recovered from `local_graph`'s node count and `x`'s flat
/// size when reshaping `x` for the call. `local_graph` is always
/// present, unlike the node/edge weights, which follow the same
/// None-means-std::nullopt convention as the plain graph partitioner.
/// Defined in graph.cpp.
dolfinx::graph::hybrid_partition_fn
partitioner_wrap_py_to_cpp(const PythonHybridPartitionFunction& p);

/// Opaque handle wrapping a partitioning function `Fn`, bound to
/// Python as its own type -- see GraphPartitioner/GeometricPartitioner/
/// HybridPartitioner below.
template <typename Fn>
struct OpaquePartitioner
{
  Fn fn;
};

/// Opaque handles for partition_fn, geom_partition_fn and
/// hybrid_partition_fn, each bound as its own Python type. nanobind
/// cannot tell which of several std::function signatures a bare Python
/// callable is meant to satisfy, so create_mesh needs distinct types
/// to disambiguate rather than relying on argument count.
using GraphPartitioner = OpaquePartitioner<dolfinx::graph::partition_fn>;
using GeometricPartitioner
    = OpaquePartitioner<dolfinx::graph::geom_partition_fn>;
using HybridPartitioner
    = OpaquePartitioner<dolfinx::graph::hybrid_partition_fn>;

/// Opaque handle for a geometric mesh reordering function.
using GeometricReorderer = OpaquePartitioner<dolfinx::graph::reorder_geom_fn>;

/// Bind the AdjacencyList<T, U> properties and methods common to every
/// instantiation: offsets, num_nodes, equality, and length.
template <typename T, typename U>
void declare_adjacency_list_common(
    nb::class_<dolfinx::graph::AdjacencyList<T, U>>& cls)
{
  cls.def_prop_ro(
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

/// Declare AdjacencyList class with __init__ methods for a given type
/// @param m The nanobind module
/// @param type String representation of the type (e.g., "int32", "int64")
template <typename T, typename U>
void declare_adjacency_list_init(nanobind::module_& m, std::string type)
{
  namespace nb = nanobind;

  std::string pyclass_name = std::string("AdjacencyList_") + type;
  nb::class_<dolfinx::graph::AdjacencyList<T, U>> cls(m, pyclass_name.c_str(),
                                                      "Adjacency List");
  cls.def(
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
      .def("__repr__", &dolfinx::graph::AdjacencyList<T, U>::str);
  declare_adjacency_list_common(cls);
}

/// Declare additional AdjacencyList properties for a given type
/// @param m The nanobind module
/// @param type String representation of the type
template <typename T, typename U>
void declare_adjacency_list(nanobind::module_& m, std::string type)
{
  std::string pyclass_name = std::string("AdjacencyList_") + type;
  nb::class_<dolfinx::graph::AdjacencyList<T, U>> cls(m, pyclass_name.c_str(),
                                                      "Adjacency List");
  declare_adjacency_list_common(cls);
}

} // namespace dolfinx_wrappers
