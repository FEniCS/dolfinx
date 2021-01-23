// Copyright (C) 2017 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "array.h"
#include "caster_mpi.h"
#include <Eigen/Core>
#include <dolfinx/geometry/BoundingBoxTree.h>
#include <dolfinx/geometry/gjk.h>
#include <dolfinx/geometry/utils.h>
#include <dolfinx/mesh/Mesh.h>
#include <memory>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace dolfinx_wrappers
{
void geometry(py::module& m)
{
  m.def("create_midpoint_tree", &dolfinx::geometry::create_midpoint_tree);
  m.def("compute_closest_entity", &dolfinx::geometry::compute_closest_entity,
        py::arg("tree"), py::arg("p"), py::arg("mesh"), py::arg("R") = -1);

  m.def("compute_collisions_point",
        py::overload_cast<const dolfinx::geometry::BoundingBoxTree&,
                          const Eigen::Vector3d&>(
            &dolfinx::geometry::compute_collisions));
  m.def("compute_collisions",
        py::overload_cast<const dolfinx::geometry::BoundingBoxTree&,
                          const dolfinx::geometry::BoundingBoxTree&>(
            &dolfinx::geometry::compute_collisions));

  m.def("compute_distance_gjk", &dolfinx::geometry::compute_distance_gjk);
  m.def("squared_distance", &dolfinx::geometry::squared_distance);
  m.def("select_colliding_cells",
        [](const dolfinx::mesh::Mesh& mesh,
           const py::array_t<std::int32_t, py::array::c_style>& candidate_cells,
           const Eigen::Vector3d& point, int n) {
          return as_pyarray(dolfinx::geometry::select_colliding_cells(
              mesh, tcb::span(candidate_cells.data(), candidate_cells.size()),
              point, n));
        });

  // dolfinx::geometry::BoundingBoxTree
  py::class_<dolfinx::geometry::BoundingBoxTree,
             std::shared_ptr<dolfinx::geometry::BoundingBoxTree>>(
      m, "BoundingBoxTree")
      .def(py::init<const dolfinx::mesh::Mesh&, int, double>(), py::arg("mesh"),
           py::arg("tdim"), py::arg("padding") = 0)
      .def(py::init<const dolfinx::mesh::Mesh&, int,
                    const std::vector<std::int32_t>&, double>(),
           py::arg("mesh"), py::arg("tdim"), py::arg("entity_indices"),
           py::arg("padding") = 0)
      .def(py::init<const std::vector<Eigen::Vector3d>&>())
      .def("num_bboxes", &dolfinx::geometry::BoundingBoxTree::num_bboxes)
      .def("get_bbox", &dolfinx::geometry::BoundingBoxTree::get_bbox)
      .def("str", &dolfinx::geometry::BoundingBoxTree::str)
      .def("compute_global_tree",
           [](const dolfinx::geometry::BoundingBoxTree& self,
              const MPICommWrapper comm) {
             return self.compute_global_tree(comm.get());
           });
}
} // namespace dolfinx_wrappers
