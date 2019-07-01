// Copyright (C) 2017 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <Eigen/Dense>
#include <dolfin/geometry/BoundingBoxTree.h>
#include <dolfin/geometry/CollisionPredicates.h>
#include <dolfin/mesh/Mesh.h>
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
  // dolfin::geometry::BoundingBoxTree
  py::class_<dolfin::geometry::BoundingBoxTree,
             std::shared_ptr<dolfin::geometry::BoundingBoxTree>>(
      m, "BoundingBoxTree")
      // .def(py::init<std::size_t>())
      .def(py::init<const dolfin::mesh::Mesh&, std::size_t>())
      .def(py::init<const std::vector<Eigen::Vector3d>&, std::size_t>())
      .def("compute_collisions",
           (std::vector<unsigned int>(dolfin::geometry::BoundingBoxTree::*)(
               const Eigen::Vector3d&) const)
               & dolfin::geometry::BoundingBoxTree::compute_collisions)
      .def("compute_collisions",
           (std::pair<std::vector<unsigned int>, std::vector<unsigned int>>(
               dolfin::geometry::BoundingBoxTree::*)(
               const dolfin::geometry::BoundingBoxTree&) const)
               & dolfin::geometry::BoundingBoxTree::compute_collisions)
      .def("compute_entity_collisions",
           (std::vector<unsigned int>(dolfin::geometry::BoundingBoxTree::*)(
               const Eigen::Vector3d&, const dolfin::mesh::Mesh&) const)
               & dolfin::geometry::BoundingBoxTree::compute_entity_collisions)
      .def("compute_entity_collisions",
           (std::pair<std::vector<unsigned int>, std::vector<unsigned int>>(
               dolfin::geometry::BoundingBoxTree::*)(
               const dolfin::geometry::BoundingBoxTree&,
               const dolfin::mesh::Mesh&, const dolfin::mesh::Mesh&) const)
               & dolfin::geometry::BoundingBoxTree::compute_entity_collisions)
      .def("compute_first_collision",
           &dolfin::geometry::BoundingBoxTree::compute_first_collision)
      .def("collides",
           &dolfin::geometry::BoundingBoxTree::collides)
      .def("compute_first_entity_collision",
           &dolfin::geometry::BoundingBoxTree::compute_first_entity_collision)
      .def("compute_closest_entity",
           &dolfin::geometry::BoundingBoxTree::compute_closest_entity)
      .def("str", &dolfin::geometry::BoundingBoxTree::str);

  // These classes are wrapped only to be able to write tests in python.
  // They are not imported into the dolfin namespace in python, but must be
  // accessed through
  // dolfin.cpp.geometry
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
