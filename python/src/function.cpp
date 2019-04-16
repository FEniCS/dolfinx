// Copyright (C) 2017 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "casters.h"
#include <cstdint>
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/function/Expression.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/function/SpecialFunctions.h>
#include <dolfin/geometry/BoundingBoxTree.h>
#include <dolfin/geometry/Point.h>
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

  // Create dolfin::function::Expression from a JIT pointer
  m.def("make_dolfin_expression",
        [](std::uintptr_t addr) {
          dolfin::function::Expression* p
              = reinterpret_cast<dolfin::function::Expression*>(addr);
          return std::shared_ptr<const dolfin::function::Expression>(p);
        },
        "Create a dolfin::function::Expression object from a pointer integer, "
        "typically returned by a just-in-time compiler");

  // dolfin:Expression
  py::class_<dolfin::function::Expression,
             std::shared_ptr<dolfin::function::Expression>>(m, "Expression")
      .def(py::init([](std::uintptr_t addr,
                       std::vector<std::size_t> value_size) {
        std::function<void(PetscScalar*, const double*, const std::int64_t*,
                           int, int, int, int)>
            f = reinterpret_cast<void (*)(PetscScalar*, const double*,
                                          const std::int64_t*, int, int, int,
                                          int)>(addr);
        return std::make_unique<dolfin::function::Expression>(f, value_size);
      }))
      .def(
          py::init<std::function<void(PetscScalar*, const double*,
                                      const std::int64_t*, int, int, int, int)>,
                   std::vector<std::size_t>>())
      .def("eval", &dolfin::function::Expression::eval)
      .def("value_dimension", &dolfin::function::Expression::value_dimension);

  // dolfin::function::Function
  py::class_<dolfin::function::Function,
             std::shared_ptr<dolfin::function::Function>,
             dolfin::common::Variable>(m, "Function",
                                       "A finite element function")
      .def(py::init<std::shared_ptr<const dolfin::function::FunctionSpace>>(),
           "Create a function on the given function space")
      .def(py::init<std::shared_ptr<dolfin::function::FunctionSpace>, Vec>())
      .def("sub", &dolfin::function::Function::sub,
           "Return sub-function (view into parent Function")
      .def("collapse",
           [](dolfin::function::Function& self) {
             return std::make_shared<dolfin::function::Function>(self);
           },
           "Collapse sub-function view.")
      .def("interpolate",
           py::overload_cast<const dolfin::function::Function&>(
               &dolfin::function::Function::interpolate),
           py::arg("u"))
      .def("interpolate",
           py::overload_cast<const dolfin::function::Expression&>(
               &dolfin::function::Function::interpolate),
           py::arg("expr"))
      .def("vector",
           [](const dolfin::function::Function& self) {
             return self.vector().vec();
           },
           "Return the vector associated with the finite element Function")
      .def("value_dimension", &dolfin::function::Function::value_dimension)
      .def("value_size", &dolfin::function::Function::value_size)
      .def("value_rank", &dolfin::function::Function::value_rank)
      .def_property_readonly("value_shape",
                             &dolfin::function::Function::value_shape)
      .def("eval",
           [](const dolfin::function::Function& self,
              Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic,
                                      Eigen::Dynamic, Eigen::RowMajor>>
                  u,
              const Eigen::Ref<const dolfin::EigenRowArrayXXd> x,
              const dolfin::mesh::Cell& cell) { self.eval(u, x, cell); },
           "Evaluate Function (cell version)")
      .def("eval",
           py::overload_cast<
               Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic,
                                       Eigen::Dynamic, Eigen::RowMajor>>,
               const Eigen::Ref<const dolfin::EigenRowArrayXXd>,
               const dolfin::geometry::BoundingBoxTree&>(
               &dolfin::function::Function::eval, py::const_),
           py::arg("values"), py::arg("x"), py::arg("bb_tree"),
           "Evaluate Function")
      .def("compute_point_values",
           py::overload_cast<const dolfin::mesh::Mesh&>(
               &dolfin::function::Function::compute_point_values, py::const_),
           "Compute values at all mesh points")
      .def("compute_point_values",
           [](dolfin::function::Function& self) {
             auto V = self.function_space();
             if (!V)
               throw py::value_error("Function has no function space. "
                                     "You must supply a mesh.");
             auto mesh = V->mesh();
             if (!mesh)
               throw py::value_error("Function has no function space "
                                     "mesh. You must supply a mesh.");
             return self.compute_point_values(*mesh);
           },
           "Compute values at all mesh points by using the mesh "
           "function.function_space().mesh()")
      .def("function_space", &dolfin::function::Function::function_space);

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
             std::shared_ptr<dolfin::function::FunctionSpace>,
             dolfin::common::Variable>(m, "FunctionSpace", py::dynamic_attr())
      .def(py::init<std::shared_ptr<dolfin::mesh::Mesh>,
                    std::shared_ptr<dolfin::fem::FiniteElement>,
                    std::shared_ptr<dolfin::fem::GenericDofMap>>())
      .def(py::init<const dolfin::function::FunctionSpace&>())
      .def("__eq__", &dolfin::function::FunctionSpace::operator==)
      .def("dim", &dolfin::function::FunctionSpace::dim)
      .def("collapse", &dolfin::function::FunctionSpace::collapse)
      .def("component", &dolfin::function::FunctionSpace::component)
      .def("contains", &dolfin::function::FunctionSpace::contains)
      .def("element", &dolfin::function::FunctionSpace::element)
      .def("mesh", &dolfin::function::FunctionSpace::mesh)
      .def("dofmap", &dolfin::function::FunctionSpace::dofmap)
      .def("set_x", &dolfin::function::FunctionSpace::set_x)
      .def("sub", &dolfin::function::FunctionSpace::sub)
      .def("tabulate_dof_coordinates",
           &dolfin::function::FunctionSpace::tabulate_dof_coordinates);
}
} // namespace dolfin_wrappers
