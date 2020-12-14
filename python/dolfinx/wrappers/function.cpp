// Copyright (C) 2017 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "caster_petsc.h"
#include <cstdint>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/fem/Expression.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/fem/interpolate.h>
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
  // dolfinx::fem::Function
  py::class_<dolfinx::fem::Function<PetscScalar>,
             std::shared_ptr<dolfinx::fem::Function<PetscScalar>>>(
      m, "Function", "A finite element function")
      .def(py::init<std::shared_ptr<const dolfinx::fem::FunctionSpace>>(),
           "Create a function on the given function space")
      .def(py::init<std::shared_ptr<dolfinx::fem::FunctionSpace>,
                    std::shared_ptr<dolfinx::la::Vector<PetscScalar>>>())
      .def_readwrite("name", &dolfinx::fem::Function<PetscScalar>::name)
      .def_property_readonly("id",
                             &dolfinx::fem::Function<PetscScalar>::id)
      .def("sub", &dolfinx::fem::Function<PetscScalar>::sub,
           "Return sub-function (view into parent Function")
      .def("collapse", &dolfinx::fem::Function<PetscScalar>::collapse,
           "Collapse sub-function view")
      .def("interpolate",
           py::overload_cast<const std::function<Eigen::Array<
               PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>(
               const Eigen::Ref<const Eigen::Array<double, 3, Eigen::Dynamic,
                                                   Eigen::RowMajor>>&)>&>(
               &dolfinx::fem::Function<PetscScalar>::interpolate),
           py::arg("f"), "Interpolate an expression")
      .def("interpolate",
           py::overload_cast<const dolfinx::fem::Function<PetscScalar>&>(
               &dolfinx::fem::Function<PetscScalar>::interpolate),
           py::arg("u"), "Interpolate a finite element function")
      .def("interpolate_ptr",
           [](dolfinx::fem::Function<PetscScalar>& self,
              std::uintptr_t addr) {
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
             dolfinx::fem::interpolate_c<PetscScalar>(self, _f);
           },
           "Interpolate using a pointer to an expression with a C signature")
      .def_property_readonly(
          "vector",
          [](const dolfinx::fem::Function<PetscScalar>&
                 self) { return self.vector(); },
          "Return the vector associated with the finite element Function")
      .def_property_readonly(
          "x",
          py::overload_cast<>(&dolfinx::fem::Function<PetscScalar>::x),
          "Return the vector associated with the finite element Function")
      .def("eval", &dolfinx::fem::Function<PetscScalar>::eval,
           py::arg("x"), py::arg("cells"), py::arg("values"),
           "Evaluate Function")
      .def("compute_point_values",
           &dolfinx::fem::Function<PetscScalar>::compute_point_values,
           "Compute values at all mesh points")
      .def_property_readonly(
          "function_space",
          &dolfinx::fem::Function<PetscScalar>::function_space);

  // dolfinx::fem::FunctionSpace
  py::class_<dolfinx::fem::FunctionSpace,
             std::shared_ptr<dolfinx::fem::FunctionSpace>>(
      m, "FunctionSpace", py::dynamic_attr())
      .def(py::init<std::shared_ptr<dolfinx::mesh::Mesh>,
                    std::shared_ptr<dolfinx::fem::FiniteElement>,
                    std::shared_ptr<dolfinx::fem::DofMap>>())
      .def_property_readonly("id", &dolfinx::fem::FunctionSpace::id)
      .def("__hash__", &dolfinx::fem::FunctionSpace::id)
      .def("__eq__", &dolfinx::fem::FunctionSpace::operator==)
      .def_property_readonly("dim", &dolfinx::fem::FunctionSpace::dim)
      .def("collapse", &dolfinx::fem::FunctionSpace::collapse)
      .def("component", &dolfinx::fem::FunctionSpace::component)
      .def("contains", &dolfinx::fem::FunctionSpace::contains)
      .def_property_readonly("element",
                             &dolfinx::fem::FunctionSpace::element)
      .def_property_readonly("mesh", &dolfinx::fem::FunctionSpace::mesh)
      .def_property_readonly("dofmap",
                             &dolfinx::fem::FunctionSpace::dofmap)
      .def("sub", &dolfinx::fem::FunctionSpace::sub)
      .def("tabulate_dof_coordinates",
           &dolfinx::fem::FunctionSpace::tabulate_dof_coordinates);

  // dolfinx::fem::Constant
  py::class_<dolfinx::fem::Constant<PetscScalar>,
             std::shared_ptr<dolfinx::fem::Constant<PetscScalar>>>(
      m, "Constant", "A value constant with respect to integration domain")
      .def(py::init<std::vector<int>, std::vector<PetscScalar>>(),
           "Create a constant from a scalar value array")
      .def("value",
           [](dolfinx::fem::Constant<PetscScalar>& self) {
             return py::array(self.shape, self.value.data(), py::none());
           },
           py::return_value_policy::reference_internal);

  // dolfinx::fem::Expression
  py::class_<dolfinx::fem::Expression<PetscScalar>,
             std::shared_ptr<dolfinx::fem::Expression<PetscScalar>>>(
      m, "Expression", "An Expression")
      .def(py::init([](
               const std::vector<std::shared_ptr<const dolfinx::fem::Function<PetscScalar>>>& coefficients,
	       const std::vector<std::shared_ptr<const dolfinx::fem::Constant<PetscScalar>>>& constants,
	       const std::shared_ptr<const dolfinx::mesh::Mesh>& mesh,
               const Eigen::Ref<const Eigen::Array<
                   double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& x,
	       py::object addr,
               const std::size_t value_size) { auto tabulate_expression_ptr = (void (*)(
                 PetscScalar*, const PetscScalar*, const PetscScalar*,
                 const double*))addr.cast<std::uintptr_t>(); return dolfinx::fem::Expression<PetscScalar>(coefficients, constants, mesh, x, tabulate_expression_ptr, value_size); }),
	       py::arg("coefficients"), py::arg("constants"), py::arg("mesh"), py::arg("x"), py::arg("fn"), py::arg("value_size"))
      .def("eval", &dolfinx::fem::Expression<PetscScalar>::eval)
      .def_property_readonly("mesh",
                             &dolfinx::fem::Expression<PetscScalar>::mesh,
                             py::return_value_policy::reference_internal)
      .def_property_readonly(
          "num_points", &dolfinx::fem::Expression<PetscScalar>::num_points,
          py::return_value_policy::reference_internal)
      .def_property_readonly(
          "value_size", &dolfinx::fem::Expression<PetscScalar>::value_size,
          py::return_value_policy::reference_internal)
      .def_property_readonly("x",
                             &dolfinx::fem::Expression<PetscScalar>::x,
                             py::return_value_policy::reference_internal);
}
} // namespace dolfinx_wrappers
