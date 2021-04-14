// Copyright (C) 2017 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "array.h"
#include "caster_mpi.h"
#include <dolfinx/geometry/BoundingBoxTree.h>
#include <dolfinx/geometry/gjk.h>
#include <dolfinx/geometry/utils.h>
#include <dolfinx/mesh/Mesh.h>
#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <xtl/xspan.hpp>

namespace py = pybind11;

namespace dolfinx_wrappers
{
void geometry(py::module& m)
{
  m.def("create_midpoint_tree", &dolfinx::geometry::create_midpoint_tree);
  m.def("compute_closest_entity", &dolfinx::geometry::compute_closest_entity,
        py::arg("tree"), py::arg("p"), py::arg("mesh"), py::arg("R") = -1);

  m.def("compute_collisions_point",
        [](const dolfinx::geometry::BoundingBoxTree& tree,
           const std::array<double, 3>& point) {
          return as_pyarray(dolfinx::geometry::compute_collisions(tree, point));
        });
  m.def("compute_collisions",
        py::overload_cast<const dolfinx::geometry::BoundingBoxTree&,
                          const dolfinx::geometry::BoundingBoxTree&>(
            &dolfinx::geometry::compute_collisions));

  m.def("compute_distance_gjk",
        [](const py::array_t<double>& p, const py::array_t<double>& q) {
          const std::size_t p_s0 = p.ndim() == 1 ? 1 : p.shape(0);
          const std::size_t q_s0 = q.ndim() == 1 ? 1 : q.shape(0);
          dolfinx::array2d<double> _p(p_s0, 3, 0.0), _q(q_s0, 3, 0.0);

          auto px = p.unchecked();
          if (px.ndim() == 1)
          {
            for (py::ssize_t i = 0; i < px.shape(0); i++)
              _p(0, i) = px(i);
          }
          else if (px.ndim() == 2)
          {
            for (py::ssize_t i = 0; i < px.shape(0); i++)
              for (py::ssize_t j = 0; j < px.shape(1); j++)
                _p(i, j) = px(i, j);
          }
          else
            throw std::runtime_error("Array has wrong ndim.");

          auto qx = q.unchecked();
          if (qx.ndim() == 1)
          {
            for (py::ssize_t i = 0; i < qx.shape(0); i++)
              _q(0, i) = qx(i);
          }
          else if (qx.ndim() == 2)
          {
            for (py::ssize_t i = 0; i < qx.shape(0); i++)
              for (py::ssize_t j = 0; j < qx.shape(1); j++)
                _q(i, j) = qx(i, j);
          }
          else
            throw std::runtime_error("Array has wrong ndim.");

          return dolfinx::geometry::compute_distance_gjk(_p, _q);
        });

  m.def("squared_distance", &dolfinx::geometry::squared_distance);
  m.def("select_colliding_cells",
        [](const dolfinx::mesh::Mesh& mesh,
           const py::array_t<std::int32_t, py::array::c_style>& candidate_cells,
           const std::array<double, 3>& point, int n) {
          return as_pyarray(dolfinx::geometry::select_colliding_cells(
              mesh,
              xtl::span<const std::int32_t>(candidate_cells.data(),
                                            candidate_cells.size()),
              point, n));
        });

  // dolfinx::geometry::BoundingBoxTree
  py::class_<dolfinx::geometry::BoundingBoxTree,
             std::shared_ptr<dolfinx::geometry::BoundingBoxTree>>(
      m, "BoundingBoxTree")
      .def(py::init<const dolfinx::mesh::Mesh&, int, double>(), py::arg("mesh"),
           py::arg("tdim"), py::arg("padding") = 0.0)
      .def(py::init(
               [](const dolfinx::mesh::Mesh& mesh, int tdim,
                  const py::array_t<std::int32_t, py::array::c_style>& entities,
                  double padding) {
                 return dolfinx::geometry::BoundingBoxTree(
                     mesh, tdim,
                     xtl::span<const std::int32_t>(entities.data(),
                                                   entities.size()),
                     padding);
               }),
           py::arg("mesh"), py::arg("tdim"), py::arg("entity_indices"),
           py::arg("padding") = 0.0)
      .def_property_readonly("num_bboxes",
                             &dolfinx::geometry::BoundingBoxTree::num_bboxes)
      .def("get_bbox", &dolfinx::geometry::BoundingBoxTree::get_bbox)
      .def("__repr__", &dolfinx::geometry::BoundingBoxTree::str)
      .def("create_global_tree",
           [](const dolfinx::geometry::BoundingBoxTree& self,
              const MPICommWrapper comm) {
             return self.create_global_tree(comm.get());
           });
}
} // namespace dolfinx_wrappers
