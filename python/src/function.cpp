// Copyright (C) 2017 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "caster_petsc.h"
#include <cstdint>
#include <dolfin/fem/DofMap.h>
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/function/Constant.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/geometry/BoundingBoxTree.h>
#include <dolfin/la/PETScVector.h>
#include <dolfin/mesh/Mesh.h>
#include <memory>
#include <petsc4py/petsc4py.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace dolfin_wrappers
{

void function(py::module& m)
{
  // dolfin::function::Function
  py::class_<dolfin::function::Function,
             std::shared_ptr<dolfin::function::Function>>(
      m, "Function", "A finite element function")
      .def(py::init<std::shared_ptr<const dolfin::function::FunctionSpace>>(),
           "Create a function on the given function space")
      .def(py::init<std::shared_ptr<dolfin::function::FunctionSpace>, Vec>())
      .def_readwrite("name", &dolfin::function::Function::name)
      .def_property_readonly("id", &dolfin::function::Function::id)
      .def("sub", &dolfin::function::Function::sub,
           "Return sub-function (view into parent Function")
      .def("collapse", &dolfin::function::Function::collapse,
           "Collapse sub-function view")
      .def(
          "interpolate",
          py::overload_cast<const std::function<void(
              Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic,
                                      Eigen::Dynamic, Eigen::RowMajor>>,
              const Eigen::Ref<const Eigen::Array<
                  double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>&)>&>(
              &dolfin::function::Function::interpolate),
          py::arg("f"), "Interpolate a function expression")
      .def("interpolate",
           py::overload_cast<const dolfin::function::Function&>(
               &dolfin::function::Function::interpolate),
           py::arg("u"), "Interpolate a finite element function")
      .def("interpolate_ptr",
           [](dolfin::function::Function& self, std::uintptr_t addr) {
             const std::function<void(PetscScalar*, int, int, const double*,
                                      int)>
                 f = reinterpret_cast<void (*)(PetscScalar*, int, int,
                                               const double*, int)>(addr);
             auto _f =
                 [&f](Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic,
                                              Eigen::Dynamic, Eigen::RowMajor>>
                          values,
                      const Eigen::Ref<
                          const Eigen::Array<double, Eigen::Dynamic,
                                             Eigen::Dynamic, Eigen::RowMajor>>&
                          x) {
                   f(values.data(), values.rows(), values.cols(), x.data(),
                     x.cols());
                 };

             self.interpolate(_f);
           },
           "Interpolate using a pointer to an expression with a C signature")
      .def_property_readonly(
          "vector",
          [](const dolfin::function::Function&
                 self) { return self.vector().vec(); },
          "Return the vector associated with the finite element Function")
      .def("value_dimension", &dolfin::function::Function::value_dimension)
      .def_property_readonly("value_size",
                             &dolfin::function::Function::value_size)
      .def_property_readonly("value_rank",
                             &dolfin::function::Function::value_rank)
      .def_property_readonly("value_shape",
                             &dolfin::function::Function::value_shape)
      .def("eval", &dolfin::function::Function::eval, py::arg("x"),
           py::arg("cells"), py::arg("values"), "Evaluate Function")
      .def("compute_point_values",
           &dolfin::function::Function::compute_point_values,
           "Compute values at all mesh points")
      .def_property_readonly("function_space",
                             &dolfin::function::Function::function_space);

  // FIXME: why is this floating here?
  m.def("interpolate",
        [](const dolfin::function::Function& f,
           std::shared_ptr<const dolfin::function::FunctionSpace> V) {
          auto g = std::make_unique<dolfin::function::Function>(V);
          g->interpolate(f);
          return g;
        });

  // dolfin::function::FunctionSpace
  py::class_<dolfin::function::FunctionSpace,
             std::shared_ptr<dolfin::function::FunctionSpace>>(
      m, "FunctionSpace", py::dynamic_attr())
      .def(py::init<std::shared_ptr<dolfin::mesh::Mesh>,
                    std::shared_ptr<dolfin::fem::FiniteElement>,
                    std::shared_ptr<dolfin::fem::DofMap>>())
      .def_property_readonly("id", &dolfin::function::FunctionSpace::id)
      .def("__eq__", &dolfin::function::FunctionSpace::operator==)
      .def("dim", &dolfin::function::FunctionSpace::dim)
      .def("collapse", &dolfin::function::FunctionSpace::collapse)
      .def("component", &dolfin::function::FunctionSpace::component)
      .def("contains", &dolfin::function::FunctionSpace::contains)
      .def_property_readonly("element",
                             &dolfin::function::FunctionSpace::element)
      .def_property_readonly("mesh", &dolfin::function::FunctionSpace::mesh)
      .def_property_readonly("dofmap", &dolfin::function::FunctionSpace::dofmap)
      .def("set_x", &dolfin::function::FunctionSpace::set_x)
      .def("sub", &dolfin::function::FunctionSpace::sub)
      .def("tabulate_dof_coordinates",
           &dolfin::function::FunctionSpace::tabulate_dof_coordinates);

  // dolfin::function::Constant
  py::class_<dolfin::function::Constant,
             std::shared_ptr<dolfin::function::Constant>>(
      m, "Constant", "A value constant with respect to integration domain")
      .def(py::init<std::vector<int>, std::vector<PetscScalar>>(),
           "Create a constant from a scalar value array")
      .def("value",
           [](dolfin::function::Constant& self) {
             return py::array(self.shape, self.value.data(), py::none());
           },
           py::return_value_policy::reference_internal);
}
} // namespace dolfin_wrappers
