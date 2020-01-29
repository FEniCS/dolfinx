// Copyright (C) 2017 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "caster_petsc.h"
#include <cstdint>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/function/Constant.h>
#include <dolfinx/function/Function.h>
#include <dolfinx/function/FunctionSpace.h>
#include <dolfinx/geometry/BoundingBoxTree.h>
#include <dolfinx/la/PETScVector.h>
#include <dolfinx/mesh/Mesh.h>
#include <memory>
#include <petsc4py/petsc4py.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace dolfinx_wrappers
{

void function(py::module& m)
{
  // dolfinx::function::Function
  py::class_<dolfinx::function::Function,
             std::shared_ptr<dolfinx::function::Function>>(
      m, "Function", "A finite element function")
      .def(py::init<std::shared_ptr<const dolfinx::function::FunctionSpace>>(),
           "Create a function on the given function space")
      .def(py::init<std::shared_ptr<dolfinx::function::FunctionSpace>, Vec>())
      .def_readwrite("name", &dolfinx::function::Function::name)
      .def_property_readonly("id", &dolfinx::function::Function::id)
      .def("sub", &dolfinx::function::Function::sub,
           "Return sub-function (view into parent Function")
      .def("collapse", &dolfinx::function::Function::collapse,
           "Collapse sub-function view")
      .def("interpolate",
           py::overload_cast<const std::function<Eigen::Array<
               PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>(
               const Eigen::Ref<const Eigen::Array<double, 3, Eigen::Dynamic,
                                                   Eigen::RowMajor>>&)>&>(
               &dolfinx::function::Function::interpolate),
           py::arg("f"), "Interpolate an expression")
      .def("interpolate",
           py::overload_cast<const dolfinx::function::Function&>(
               &dolfinx::function::Function::interpolate),
           py::arg("u"), "Interpolate a finite element function")
      .def("interpolate_ptr",
           [](dolfinx::function::Function& self, std::uintptr_t addr) {
             const std::function<void(PetscScalar*, int, int, const double*)> f
                 = reinterpret_cast<void (*)(PetscScalar*, int, int,
                                             const double*)>(addr);
             auto _f =
                 [&f](Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic,
                                              Eigen::Dynamic, Eigen::RowMajor>>
                          values,
                      const Eigen::Ref<const Eigen::Array<
                          double, Eigen::Dynamic, 3, Eigen::RowMajor>>& x) {
                   f(values.data(), values.rows(), values.cols(), x.data());
                 };

             self.interpolate_c(_f);
           },
           "Interpolate using a pointer to an expression with a C signature")
      .def_property_readonly(
          "vector",
          [](const dolfinx::function::Function&
                 self) { return self.vector().vec(); },
          "Return the vector associated with the finite element Function")
      .def("value_dimension", &dolfinx::function::Function::value_dimension)
      .def_property_readonly("value_size",
                             &dolfinx::function::Function::value_size)
      .def_property_readonly("value_rank",
                             &dolfinx::function::Function::value_rank)
      .def_property_readonly("value_shape",
                             &dolfinx::function::Function::value_shape)
      .def("eval", &dolfinx::function::Function::eval, py::arg("x"),
           py::arg("cells"), py::arg("values"), "Evaluate Function")
      .def("compute_point_values",
           &dolfinx::function::Function::compute_point_values,
           "Compute values at all mesh points")
      .def_property_readonly("function_space",
                             &dolfinx::function::Function::function_space);

  // dolfinx::function::FunctionSpace
  py::class_<dolfinx::function::FunctionSpace,
             std::shared_ptr<dolfinx::function::FunctionSpace>>(
      m, "FunctionSpace", py::dynamic_attr())
      .def(py::init<std::shared_ptr<dolfinx::mesh::Mesh>,
                    std::shared_ptr<dolfinx::fem::FiniteElement>,
                    std::shared_ptr<dolfinx::fem::DofMap>>())
      .def_property_readonly("id", &dolfinx::function::FunctionSpace::id)
      .def("__eq__", &dolfinx::function::FunctionSpace::operator==)
      .def("dim", &dolfinx::function::FunctionSpace::dim)
      .def("collapse", &dolfinx::function::FunctionSpace::collapse)
      .def("component", &dolfinx::function::FunctionSpace::component)
      .def("contains", &dolfinx::function::FunctionSpace::contains)
      .def_property_readonly("element",
                             &dolfinx::function::FunctionSpace::element)
      .def_property_readonly("mesh", &dolfinx::function::FunctionSpace::mesh)
      .def_property_readonly("dofmap", &dolfinx::function::FunctionSpace::dofmap)
      .def("set_x", &dolfinx::function::FunctionSpace::set_x)
      .def("sub", &dolfinx::function::FunctionSpace::sub)
      .def("tabulate_dof_coordinates",
           &dolfinx::function::FunctionSpace::tabulate_dof_coordinates);

  // dolfinx::function::Constant
  py::class_<dolfinx::function::Constant,
             std::shared_ptr<dolfinx::function::Constant>>(
      m, "Constant", "A value constant with respect to integration domain")
      .def(py::init<std::vector<int>, std::vector<PetscScalar>>(),
           "Create a constant from a scalar value array")
      .def("value",
           [](dolfinx::function::Constant& self) {
             return py::array(self.shape, self.value.data(), py::none());
           },
           py::return_value_policy::reference_internal);
}
} // namespace dolfinx_wrappers
