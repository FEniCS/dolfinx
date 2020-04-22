// Copyright (C) 2017 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <Eigen/Dense>
#include <dolfinx/geometry/BoundingBoxTree.h>
#include <dolfinx/geometry/CollisionPredicates.h>
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
  m.def("compute_first_entity_collision",
        [](const dolfinx::geometry::BoundingBoxTree& tree,
           const dolfinx::mesh::Mesh& mesh,
           const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 3,
                                               Eigen::RowMajor>>& p) {
          Eigen::VectorXi entities(p.rows());
          for (Eigen::Index i = 0; i < p.rows(); ++i)
          {
            entities(i) = dolfinx::geometry::compute_first_entity_collision(
                tree, p.row(i).transpose(), mesh);
          }
          return entities;
        });
  m.def("compute_collisions_point",
        [](const dolfinx::geometry::BoundingBoxTree& tree,
           const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 3,
                                               Eigen::RowMajor>>& p) {
          std::vector<int> entities;
          std::vector<int> offset(p.rows() + 1, 0);
          for (Eigen::Index i = 0; i < p.rows(); ++i)
          {
            const std::vector<int> collisions
                = dolfinx::geometry::compute_collisions(tree,
                                                        p.row(i).transpose());
            entities.insert(entities.end(), collisions.begin(),
                            collisions.end());
            offset[i + 1] = offset[i + 1] + collisions.size();
          }
          return py::make_tuple(
              py::array_t<int>(entities.size(), entities.data()),
              py::array_t<int>(offset.size(), offset.data()));
        });
  m.def("compute_collisions",
        py::overload_cast<const dolfinx::geometry::BoundingBoxTree&,
                          const dolfinx::geometry::BoundingBoxTree&>(
            &dolfinx::geometry::compute_collisions));
  m.def("compute_entity_collisions_mesh",
        [](const dolfinx::geometry::BoundingBoxTree& tree,
           const dolfinx::mesh::Mesh& mesh,
           const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 3,
                                               Eigen::RowMajor>>& p) {
          std::vector<int> entities;
          std::vector<int> offset(p.rows() + 1, 0);
          for (Eigen::Index i = 0; i < p.rows(); ++i)
          {
            const std::vector<int> collisions
                = dolfinx::geometry::compute_entity_collisions(
                    tree, p.row(i).transpose(), mesh);
            entities.insert(entities.end(), collisions.begin(),
                            collisions.end());
            offset[i + 1] = offset[i + 1] + collisions.size();
          }
          return py::make_tuple(
              py::array_t<int>(entities.size(), entities.data()),
              py::array_t<int>(offset.size(), offset.data()));
        });
  m.def(
      "compute_entity_collisions_bb",
      py::overload_cast<const dolfinx::geometry::BoundingBoxTree&,
                        const dolfinx::geometry::BoundingBoxTree&,
                        const dolfinx::mesh::Mesh&, const dolfinx::mesh::Mesh&>(
          &dolfinx::geometry::compute_entity_collisions));
  m.def("squared_distance", &dolfinx::geometry::squared_distance);

  // dolfinx::geometry::BoundingBoxTree
  py::class_<dolfinx::geometry::BoundingBoxTree,
             std::shared_ptr<dolfinx::geometry::BoundingBoxTree>>(
      m, "BoundingBoxTree")
      .def(py::init<const dolfinx::mesh::Mesh&, int>())
      .def(py::init<const std::vector<Eigen::Vector3d>&>());

  // These classes are wrapped only to be able to write tests in python.
  // They are not imported into the dolfinx namespace in python, but must
  // be accessed through dolfinx.cpp.geometry
  py::class_<dolfinx::geometry::CollisionPredicates>(m, "CollisionPredicates")
      .def_static(
          "collides_segment_point_2d",
          &dolfinx::geometry::CollisionPredicates::collides_segment_point_2d)
      .def_static(
          "collides_triangle_point_2d",
          &dolfinx::geometry::CollisionPredicates::collides_triangle_point_2d)
      .def_static("collides_triangle_triangle_2d",
                  &dolfinx::geometry::CollisionPredicates::
                      collides_triangle_triangle_2d)
      .def_static(
          "collides_segment_segment_2d",
          &dolfinx::geometry::CollisionPredicates::collides_segment_segment_2d);
}
} // namespace dolfinx_wrappers
