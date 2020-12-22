// Copyright (C) 2017 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "caster_mpi.h"
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/graph/partition.h>
#include <pybind11/eigen.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;

namespace dolfinx_wrappers
{

template <typename T>
void declare_adjacency_list(py::module& m, std::string type)
{
  std::string pyclass_name = std::string("AdjacencyList_") + type;
  py::class_<dolfinx::graph::AdjacencyList<T>,
             std::shared_ptr<dolfinx::graph::AdjacencyList<T>>>(
      m, pyclass_name.c_str(), "Adjacency List")
      .def(py::init<const Eigen::Ref<const Eigen::Array<
               T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>&>())
      .def(
          "links",
          [](const dolfinx::graph::AdjacencyList<T>& self, int i) {
            tcb::span<const T> link = self.links(i);
            return py::array_t<T>(link.size(), link.data(), py::cast(self));
          },
          "Links (edges) of a node")
      .def_property_readonly("array",
                             [](const dolfinx::graph::AdjacencyList<T>& self) {
                               return py::array_t<T>(self.array().size(),
                                                     self.array().data(),
                                                     py::cast(self));
                             })
      .def_property_readonly("offsets",
                             [](const dolfinx::graph::AdjacencyList<T>& self) {
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
}
} // namespace dolfinx_wrappers
