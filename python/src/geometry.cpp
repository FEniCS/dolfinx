// Copyright (C) 2017 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.

#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <Eigen/Dense>

#include <dolfin/geometry/intersect.h>
#include <dolfin/geometry/BoundingBoxTree.h>
#include <dolfin/geometry/MeshPointIntersection.h>
#include <dolfin/geometry/CollisionPredicates.h>
#include <dolfin/geometry/IntersectionConstruction.h>
#include <dolfin/geometry/Point.h>
#include <dolfin/mesh/Mesh.h>

namespace py = pybind11;

namespace dolfin_wrappers
{

  void geometry(py::module& m)
  {
    // dolfin::BoundingBoxTree
    py::class_<dolfin::BoundingBoxTree, std::shared_ptr<dolfin::BoundingBoxTree>>
      (m, "BoundingBoxTree")
      .def(py::init<>())
      .def("build", (void (dolfin::BoundingBoxTree::*)(const dolfin::Mesh&))
           &dolfin::BoundingBoxTree::build)
      .def("build", (void (dolfin::BoundingBoxTree::*)(const dolfin::Mesh&, std::size_t))
           &dolfin::BoundingBoxTree::build)
      .def("compute_collisions", (std::vector<unsigned int> (dolfin::BoundingBoxTree::*)(const dolfin::Point&) const)
           &dolfin::BoundingBoxTree::compute_collisions)
      .def("compute_collisions",
           (std::pair<std::vector<unsigned int>, std::vector<unsigned int>>
            (dolfin::BoundingBoxTree::*)(const dolfin::BoundingBoxTree&) const)
           &dolfin::BoundingBoxTree::compute_collisions)
      .def("compute_entity_collisions", (std::vector<unsigned int> (dolfin::BoundingBoxTree::*)(const dolfin::Point&) const)
           &dolfin::BoundingBoxTree::compute_entity_collisions)
      .def("compute_entity_collisions",
           (std::pair<std::vector<unsigned int>, std::vector<unsigned int>>
            (dolfin::BoundingBoxTree::*)(const dolfin::BoundingBoxTree&) const)
	   &dolfin::BoundingBoxTree::compute_entity_collisions)
      .def("compute_first_collision", &dolfin::BoundingBoxTree::compute_first_collision)
      .def("compute_first_entity_collision", &dolfin::BoundingBoxTree::compute_first_entity_collision)
      .def("compute_closest_entity", &dolfin::BoundingBoxTree::compute_closest_entity);

    // dolfin::Point
    py::class_<dolfin::Point>(m, "Point")
      .def(py::init<>())
      .def(py::init<double, double, double>())
      .def(py::init<double, double>())
      .def(py::init<double>())
      .def(py::init([](py::array_t<double> x)
                    {
                      auto b = x.request();
                      assert(b.shape.size() == 1);
                      assert(b.shape[0] <= 3);
                      return dolfin::Point(b.shape[0], x.data());
                    }))
      .def("__getitem__", [](dolfin::Point& self, std::size_t index)
           {
             if (index > 2)
               throw py::index_error("Out of range");
             return self[index];
           })
      .def("__getitem__", [](const dolfin::Point& instance, py::slice slice)
           {
             std::size_t start, stop, step, slicelength;
             if (!slice.compute(3, &start, &stop, &step, &slicelength))
               throw py::error_already_set();

             if (start != 0 or stop != 3 or step != 1)
               throw std::range_error("Only full slices are supported");

             return py::array_t<double>(3, instance.coordinates());
           })
      .def("__setitem__", [](dolfin::Point& self, std::size_t index, double value)
           {
             if (index > 2)
               throw py::index_error("Out of range");
             self[index] = value;
           })
      .def("__setitem__", [](dolfin::Point& instance, py::slice slice, py::array_t<double> values)
           {
             std::size_t start, stop, step, slicelength;
             if (!slice.compute(3, &start, &stop, &step, &slicelength))
               throw py::error_already_set();

             if (start != 0 or stop != 3 or step != 1)
               throw std::range_error("Only full slices are supported");

             auto b = values.request();
             if (b.ndim != 1)
               throw std::range_error("Can only assign vector to a Point");
            if (b.shape[0] != 3)
              throw std::range_error("Can only assign vector of length 3 to a Point");

            double* x = instance.coordinates();
            std::copy_n(values.data(), 3, x);
           })
      .def(py::self + py::self)
      .def(py::self - py::self)
      .def(py::self == py::self)
      .def(py::self * float())
      .def(py::self / float())
      .def("array",
           [](dolfin::Point& self) { return Eigen::Vector3d(self.coordinates()); },
           "Return copy of coordinate array")
      .def("norm", &dolfin::Point::norm)
      .def("x", &dolfin::Point::x)
      .def("y", &dolfin::Point::y)
      .def("z", &dolfin::Point::z)
      .def("distance", &dolfin::Point::distance);

    // dolfin::MeshPointIntersection
    py::class_<dolfin::MeshPointIntersection,
               std::shared_ptr<dolfin::MeshPointIntersection>>
      (m, "MeshPointIntersection")
      .def("intersected_cells", &dolfin::MeshPointIntersection::intersected_cells);

    // These classes are wrapped only to be able to write tests in python.
    // They are not imported into the dolfin namespace in python, but must be accessed through
    // dolfin.cpp.geometry
    py::class_<dolfin::CollisionPredicates>(m, "CollisionPredicates")
      .def_static("collides_segment_point_2d",
		  &dolfin::CollisionPredicates::collides_segment_point_2d)
      .def_static("collides_triangle_point_2d",
		  &dolfin::CollisionPredicates::collides_triangle_point_2d)
      .def_static("collides_triangle_triangle_2d",
		  &dolfin::CollisionPredicates::collides_triangle_triangle_2d)
      .def_static("collides_segment_segment_2d",
		  &dolfin::CollisionPredicates::collides_segment_segment_2d);

    py::class_<dolfin::IntersectionConstruction>(m, "IntersectionConstruction")
      .def_static("intersection_triangle_triangle_2d",
		  &dolfin::IntersectionConstruction::intersection_triangle_triangle_2d)
      .def_static("intersection_segment_segment_2d",
		  &dolfin::IntersectionConstruction::intersection_segment_segment_2d)
      .def_static("intersection_triangle_segment_2d",
		  &dolfin::IntersectionConstruction::intersection_triangle_segment_2d);

    // dolfin/geometry free functions
    m.def("intersect", &dolfin::intersect);

  }
}
