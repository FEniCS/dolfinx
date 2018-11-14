// Copyright (C) 2017 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <dolfin/fem/FiniteElement.h>
#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/function/Expression.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/function/SpecialFunctions.h>
#include <dolfin/geometry/Point.h>
#include <dolfin/la/PETScVector.h>
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

void function(py::module& m)
{

  // Create dolfin::function::Expression from a JIT pointer
  m.def("make_dolfin_expression",
        [](std::uintptr_t e) {
          dolfin::function::Expression* p
              = reinterpret_cast<dolfin::function::Expression*>(e);
          return std::shared_ptr<const dolfin::function::Expression>(p);
        },
        "Create a dolfin::function::Expression object from a pointer integer, "
        "typically "
        "returned by a just-in-time compiler");

  // dolfin:Expression
  py::class_<dolfin::function::Expression,
             std::shared_ptr<dolfin::function::Expression>>(m, "Expression")
      .def(py::init<std::vector<std::size_t>>())
      .def("value_dimension", &dolfin::function::Expression::value_dimension)
      .def("set_eval",
           [](dolfin::function::Expression& self, std::size_t addr) {
             auto eval_ptr = (void (*)(
                 PetscScalar * values, const double* x, const int64_t* cell_idx,
                 int num_points, int value_size, int gdim, int num_cells)) addr;
             self.eval = eval_ptr;
           });

  // dolfin::function::Function
  py::class_<dolfin::function::Function,
             std::shared_ptr<dolfin::function::Function>>(
      m, "Function", "A finite element function")
      .def(py::init<std::shared_ptr<const dolfin::function::FunctionSpace>>(),
           "Create a function on the given function space")
      .def(py::init<std::shared_ptr<dolfin::function::FunctionSpace>,
                    std::shared_ptr<dolfin::la::PETScVector>>())
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
      // FIXME: A lot of error when using non-const version - misused
      // by Python interface?
      .def("vector",
           (std::shared_ptr<const dolfin::la::PETScVector>(
               dolfin::function::Function::*)() const)
               & dolfin::function::Function::vector,
           "Return the vector associated with the finite element Function");

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
