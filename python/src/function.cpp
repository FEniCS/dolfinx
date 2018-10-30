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
  // GenericFunction
  py::class_<dolfin::function::GenericFunction,
             std::shared_ptr<dolfin::function::GenericFunction>,
             dolfin::common::Variable>(m, "GenericFunction")
      .def("value_dimension",
           &dolfin::function::GenericFunction::value_dimension)
      .def("value_size", &dolfin::function::GenericFunction::value_size)
      .def("value_rank", &dolfin::function::GenericFunction::value_rank)
      .def_property_readonly("value_shape",
                             &dolfin::function::GenericFunction::value_shape)
      // FIXME: Change eval function to return NumPy array
      // FIXME: Add C++ version that takes a dolfin::mesh::Cell
      .def("eval",
           [](const dolfin::function::GenericFunction& self,
              Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic,
                                      Eigen::Dynamic, Eigen::RowMajor>>
                  u,
              const Eigen::Ref<const dolfin::EigenRowArrayXXd> x,
              const dolfin::mesh::Cell& cell) { self.eval(u, x, cell); },
           "Evaluate GenericFunction (cell version)")
      .def("eval",
           py::overload_cast<
               Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic,
                                       Eigen::Dynamic, Eigen::RowMajor>>,
               const Eigen::Ref<const dolfin::EigenRowArrayXXd>>(
               &dolfin::function::GenericFunction::eval, py::const_),
           py::arg("values"), py::arg("x"), "Evaluate GenericFunction")
      .def("compute_point_values",
           py::overload_cast<const dolfin::mesh::Mesh&>(
               &dolfin::function::GenericFunction::compute_point_values,
               py::const_),
           "Compute values at all mesh points")
      .def("compute_point_values",
           [](dolfin::function::GenericFunction& self) {
             auto V = self.function_space();
             if (!V)
               throw py::value_error("GenericFunction has no function space. "
                                     "You must supply a mesh.");
             auto mesh = V->mesh();
             if (!mesh)
               throw py::value_error("GenericFunction has no function space "
                                     "mesh. You must supply a mesh.");
             return self.compute_point_values(*mesh);
           },
           "Compute values at all mesh points by using the mesh "
           "function.function_space().mesh()")
      .def("function_space",
           &dolfin::function::GenericFunction::function_space);

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

  // dolfin::function::Expression trampoline (used for overloading virtual
  // function from Python)
  class PyExpression : public dolfin::function::Expression
  {
    using dolfin::function::Expression::Expression;

    void eval(Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic,
                                      Eigen::Dynamic, Eigen::RowMajor>>
                  values,
              Eigen::Ref<const dolfin::EigenRowArrayXXd> x) const override
    {
      PYBIND11_OVERLOAD(void, dolfin::function::Expression, eval, values, x);
    }

    void eval(Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic,
                                      Eigen::Dynamic, Eigen::RowMajor>>
                  values,
              Eigen::Ref<const dolfin::EigenRowArrayXXd> x,
              const dolfin::mesh::Cell& cell) const override
    {
      PYBIND11_OVERLOAD_NAME(void, dolfin::function::Expression, "eval_cell",
                             eval, values, x, cell);
    }
  };

  // dolfin:Expression
  py::class_<dolfin::function::Expression, PyExpression,
             std::shared_ptr<dolfin::function::Expression>,
             dolfin::function::GenericFunction>(
      m, "Expression",
      "An Expression is a function (field) that can appear as "
      "a coefficient in a form")
      .def(py::init<std::vector<std::size_t>>())
      .def("value_dimension", &dolfin::function::Expression::value_dimension)
      .def("get_property", &dolfin::function::Expression::get_property)
      .def("set_property", &dolfin::function::Expression::set_property)
      .def("set_generic_function",
           &dolfin::function::Expression::set_generic_function)
      .def("get_generic_function",
           &dolfin::function::Expression::get_generic_function);

  // dolfin::FacetArea
  py::class_<dolfin::function::FacetArea,
             std::shared_ptr<dolfin::function::FacetArea>,
             dolfin::function::Expression, dolfin::function::GenericFunction>(
      m, "FacetArea")
      .def(py::init<std::shared_ptr<const dolfin::mesh::Mesh>>());

  // dolfin::mesh::MeshCoordinates
  py::class_<dolfin::function::MeshCoordinates,
             std::shared_ptr<dolfin::function::MeshCoordinates>,
             dolfin::function::Expression, dolfin::function::GenericFunction>(
      m, "MeshCoordinates")
      .def(py::init<std::shared_ptr<const dolfin::mesh::Mesh>>());

  // dolfin::function::Function
  py::class_<dolfin::function::Function,
             std::shared_ptr<dolfin::function::Function>,
             dolfin::function::GenericFunction>(m, "Function",
                                                "A finite element function")
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
      .def("interpolate", &dolfin::function::Function::interpolate,
           "Interpolate the function u")
      // FIXME: A lot of error when using non-const version - misused
      // by Python interface?
      .def("vector",
           (std::shared_ptr<const dolfin::la::PETScVector>(
               dolfin::function::Function::*)() const)
               & dolfin::function::Function::vector,
           "Return the vector associated with the finite element Function");

  // FIXME: why is this floating here?
  m.def("interpolate",
        [](const dolfin::function::GenericFunction& f,
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
