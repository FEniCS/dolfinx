// Copyright (C) 2017 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <Eigen/Dense>
#include <dolfin/geometry/BoundingBoxTree.h>
#include <dolfin/geometry/CollisionPredicates.h>
#include <dolfin/geometry/utils.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshEntity.h>
#include <memory>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace dolfin_wrappers
{
void geometry(py::module& m)
{
  m.def("create_midpoint_tree", &dolfin::geometry::create_midpoint_tree);

  m.def("compute_closest_entity", &dolfin::geometry::compute_closest_entity);
  m.def("compute_first_collision", &dolfin::geometry::compute_first_collision);
  m.def("compute_first_entity_collision",
        &dolfin::geometry::compute_first_entity_collision);
  m.def("compute_collisions",
        py::overload_cast<const dolfin::geometry::BoundingBoxTree&,
                          const Eigen::Vector3d&>(
            &dolfin::geometry::compute_collisions));
  m.def("compute_collisions",
        py::overload_cast<const dolfin::geometry::BoundingBoxTree&,
                          const dolfin::geometry::BoundingBoxTree&>(
            &dolfin::geometry::compute_collisions));
  m.def("compute_entity_collisions",
        py::overload_cast<const dolfin::geometry::BoundingBoxTree&,
                          const Eigen::Vector3d&, const dolfin::mesh::Mesh&>(
            &dolfin::geometry::compute_entity_collisions));
  m.def("compute_entity_collisions",
        py::overload_cast<const dolfin::geometry::BoundingBoxTree&,
                          const dolfin::geometry::BoundingBoxTree&,
                          const dolfin::mesh::Mesh&, const dolfin::mesh::Mesh&>(
            &dolfin::geometry::compute_entity_collisions));
  m.def("squared_distance", &dolfin::geometry::squared_distance);

  // dolfin::geometry::BoundingBoxTree
  py::class_<dolfin::geometry::BoundingBoxTree,
             std::shared_ptr<dolfin::geometry::BoundingBoxTree>>(
      m, "BoundingBoxTree")
      .def(py::init<const dolfin::mesh::Mesh&, int>())
      .def(py::init<const std::vector<Eigen::Vector3d>&>())
      .def("str", &dolfin::geometry::BoundingBoxTree::str);

  // These classes are wrapped only to be able to write tests in python.
  // They are not imported into the dolfin namespace in python, but must
  // be accessed through dolfin.cpp.geometry
  py::class_<dolfin::geometry::CollisionPredicates>(m, "CollisionPredicates")
      .def_static(
          "collides_segment_point_2d",
          &dolfin::geometry::CollisionPredicates::collides_segment_point_2d)
      .def_static(
          "collides_triangle_point_2d",
          &dolfin::geometry::CollisionPredicates::collides_triangle_point_2d)
      .def_static(
          "collides_triangle_triangle_2d",
          &dolfin::geometry::CollisionPredicates::collides_triangle_triangle_2d)
      .def_static(
          "collides_segment_segment_2d",
          &dolfin::geometry::CollisionPredicates::collides_segment_segment_2d);
}
} // namespace dolfin_wrappers
