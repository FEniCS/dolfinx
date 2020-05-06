// Copyright (C) 2017 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <Eigen/Dense>
#include <dolfinx/geometry/BoundingBoxTree.h>
#include <dolfinx/geometry/utils.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/MeshEntity.h>
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

  m.def("compute_closest_entity",
        [](const dolfinx::geometry::BoundingBoxTree& tree,
           const dolfinx::geometry::BoundingBoxTree& tree_midpoint,
           const dolfinx::mesh::Mesh& mesh,
           const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 3,
                                               Eigen::RowMajor>>& p) {
          Eigen::VectorXi entities(p.rows());
          Eigen::VectorXd distance(p.rows());
          for (Eigen::Index i = 0; i < p.rows(); ++i)
          {
            const std::pair<int, double> out
                = dolfinx::geometry::compute_closest_entity(
                    tree, tree_midpoint, p.row(i).transpose(), mesh);
            entities(i) = out.first;
            distance(i) = out.second;
          }
          return std::make_pair(std::move(entities), std::move(distance));
        });
  m.def("compute_first_collision",
        [](const dolfinx::geometry::BoundingBoxTree& tree,
           const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 3,
                                               Eigen::RowMajor>>& p) {
          Eigen::VectorXi entities(p.rows());
          for (Eigen::Index i = 0; i < p.rows(); ++i)
          {
            entities(i) = dolfinx::geometry::compute_first_collision(
                tree, p.row(i).transpose());
          }
          return entities;
        });

  m.def("compute_collisions_point",
        [](const dolfinx::geometry::BoundingBoxTree& tree,
           const Eigen::Vector3d& p) {
          return dolfinx::geometry::compute_collisions(tree, p);
        });

  m.def("compute_collisions",
        py::overload_cast<const dolfinx::geometry::BoundingBoxTree&,
                          const dolfinx::geometry::BoundingBoxTree&>(
            &dolfinx::geometry::compute_collisions));

  m.def("squared_distance", &dolfinx::geometry::squared_distance);

  // dolfinx::geometry::BoundingBoxTree
  py::class_<dolfinx::geometry::BoundingBoxTree,
             std::shared_ptr<dolfinx::geometry::BoundingBoxTree>>(
      m, "BoundingBoxTree")
      .def(py::init<const dolfinx::mesh::Mesh&, int>())
      .def(py::init<const std::vector<Eigen::Vector3d>&>());
}
} // namespace dolfinx_wrappers