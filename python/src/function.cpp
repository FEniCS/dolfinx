/// Copyright (C) 2017 Chris N. Richardson and Garth N. Wells
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
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/operators.h>

#include <dolfin/common/Array.h>
#include <dolfin/function/assign.h>
#include <dolfin/function/Constant.h>
#include <dolfin/function/Expression.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionAssigner.h>
#include <dolfin/function/FunctionAXPY.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/function/LagrangeInterpolator.h>
#include <dolfin/function/SpecialFunctions.h>
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/geometry/Point.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/mesh/Mesh.h>

namespace py = pybind11;

namespace dolfin_wrappers
{

  void function(py::module& m)
  {
    // ufc::shape
    py::class_<ufc::shape>(m, "ufc_shape");

    // ufc::cell
    py::class_<ufc::cell, std::shared_ptr<ufc::cell>>(m, "ufc_cell")
      .def_readonly("cell_shape", &ufc::cell::cell_shape)
      .def_readonly("topological_dimension", &ufc::cell::topological_dimension)
      .def_readonly("geometric_dimension", &ufc::cell::geometric_dimension)
      .def_readonly("local_facet", &ufc::cell::local_facet)
      .def_readonly("mesh_identifier", &ufc::cell::mesh_identifier)
      .def_readonly("index", &ufc::cell::index);

    // ufc::function
    py::class_<ufc::function, std::shared_ptr<ufc::function>>(m, "ufc_function");

    // dolfin::GenericFunction
    py::class_<dolfin::GenericFunction, std::shared_ptr<dolfin::GenericFunction>,
               ufc::function, dolfin::Variable>
      (m, "GenericFunction")
      .def("value_dimension", &dolfin::GenericFunction::value_dimension)
      .def("value_size", &dolfin::GenericFunction::value_size)
      .def("value_rank", &dolfin::GenericFunction::value_rank)
      .def_property_readonly("value_shape", &dolfin::GenericFunction::value_shape)
      // FIXME: Change eval function to return NumPy array
      // FIXME: Add C++ version that takes a dolfin::Cell
      .def("eval", [](const dolfin::GenericFunction& self, Eigen::Ref<Eigen::VectorXd> u,
                      Eigen::Ref<const Eigen::VectorXd> x, const dolfin::Cell& cell)
           {
             ufc::cell ufc_cell;
             cell.get_cell_data(ufc_cell);
             self.eval(u, x, ufc_cell);
           }, "Evaluate GenericFunction (cell version)")
      .def("eval", (void (dolfin::GenericFunction::*)(Eigen::Ref<Eigen::VectorXd>,
                                                      Eigen::Ref<const Eigen::VectorXd>, const ufc::cell&) const)
           &dolfin::GenericFunction::eval,
           "Evaluate GenericFunction (cell version)")
      .def("eval", (void (dolfin::GenericFunction::*)(Eigen::Ref<Eigen::VectorXd>,
                                                      Eigen::Ref<const Eigen::VectorXd>) const)
           &dolfin::GenericFunction::eval, py::arg("values"), py::arg("x"), "Evaluate GenericFunction")
      .def("compute_vertex_values", [](dolfin::GenericFunction& self, const dolfin::Mesh& mesh)
           { std::vector<double> values;
             self.compute_vertex_values(values, mesh);
             return py::array_t<double>(values.size(), values.data());
           }, "Compute values at all mesh vertices")
      .def("compute_vertex_values", [](dolfin::GenericFunction& self)
           {
             auto V = self.function_space();
             if (!V)
                 throw py::value_error("GenericFunction has no function space. You must supply a mesh.");
             auto mesh = V->mesh();
             if (!mesh)
                 throw py::value_error("GenericFunction has no function space mesh. You must supply a mesh.");
             std::vector<double> values;
             self.compute_vertex_values(values, *mesh);
             // FIXME: this causes a copy, we should rewrite the C++ interface to use Eigen when SWIG is removed
             return py::array_t<double>(values.size(), values.data());
           }, "Compute values at all mesh vertices by using the mesh function.function_space().mesh()")
      .def("function_space", &dolfin::GenericFunction::function_space);

    // Create dolfin::Expression from a JIT pointer
    m.def("make_dolfin_expression",
          [](std::uintptr_t e)
          {
            dolfin::Expression *p = reinterpret_cast<dolfin::Expression *>(e);
            return std::shared_ptr<const dolfin::Expression>(p);
          }, "Create a dolfin::Expression object from a pointer integer, typically returned by a just-in-time compiler");

    // dolfin::Expression trampoline (used for overloading virtual
    // function from Python)
    class PyExpression : public dolfin::Expression
    {
      using dolfin::Expression::Expression;

      void eval(Eigen::Ref<Eigen::VectorXd> values,
                Eigen::Ref<const Eigen::VectorXd> x) const override
      { PYBIND11_OVERLOAD(void, dolfin::Expression, eval, values, x); }

      void eval(Eigen::Ref<Eigen::VectorXd> values,
                Eigen::Ref<const Eigen::VectorXd> x,
                const ufc::cell& cell) const override
      { PYBIND11_OVERLOAD_NAME(void, dolfin::Expression, "eval_cell", eval, values, x, cell); }

    };

    // dolfin:Expression
    py::class_<dolfin::Expression, PyExpression, std::shared_ptr<dolfin::Expression>,
               dolfin::GenericFunction>(m, "Expression", "An Expression is a function (field) that can appear as a coefficient in a form")
      .def(py::init<>())
      .def(py::init<std::size_t>())
      .def(py::init<std::size_t, std::size_t>())
      .def(py::init<std::vector<std::size_t>>())
      .def("__call__", [](const dolfin::Expression& self, Eigen::Ref<const Eigen::VectorXd> x)
           {
             Eigen::VectorXd f(self.value_size());
             self.eval(f, x);
             return f;
           })
      .def("__call__", [](const dolfin::Expression& self, const dolfin::Point& p)
           {
             const Eigen::Map<Eigen::VectorXd> x(const_cast<double*>(p.coordinates()), 3);
             Eigen::VectorXd f(self.value_size());
             self.eval(f, x);
             return f;
           })
      .def("value_dimension", &dolfin::Expression::value_dimension)
      .def("get_property", &dolfin::Expression::get_property)
      .def("set_property", [](dolfin::Expression& self, std::string name, py::object value)
           {
             if (py::isinstance<dolfin::GenericFunction>(value))
             {
               auto _v = value.cast<std::shared_ptr<dolfin::GenericFunction>>();
               self.set_generic_function(name, _v);
             }
             else if (py::hasattr(value, "_cpp_object"))
             {
               auto _v = value.attr("_cpp_object").cast<std::shared_ptr<dolfin::GenericFunction>>();
               self.set_generic_function(name, _v);
             }
             else
             {
               double _v = value.cast<double>();
               self.set_property(name, _v);
             }
           })
      .def("get_generic_function", &dolfin::Expression::get_generic_function);

    // dolfin::Constant
    py::class_<dolfin::Constant, std::shared_ptr<dolfin::Constant>, dolfin::Expression>
      (m, "Constant")
      .def(py::init<double>())
      .def(py::init<std::vector<double>>())
      .def(py::init<std::vector<std::size_t>, std::vector<double>>())
      .def("values", [](const dolfin::Constant& self)
           { auto v =  self.values(); return py::array_t<double>(v.size(), v.data()); })
      .def("__float__", [](const dolfin::Constant& instance) -> double { return instance; })
      /*
      .def("_assign", [](dolfin::Constant& self, const dolfin::Constant& other) -> const dolfin::Constant&
           {self = other;})
      .def("_assign", [](dolfin::Constant& self, double value) -> const dolfin::Constant&
           {self = value;})
      */
      .def("assign", [](dolfin::Constant& self, const dolfin::Constant& other) {self = other;})
      .def("assign", [](dolfin::Constant& self, double value) {self = value;})
      /*
      .def("_assign", (const dolfin::Constant& (dolfin::Constant::*)(const dolfin::Constant&))
                       &dolfin::Constant::operator=)
      .def("_assign", (const dolfin::Constant& (dolfin::Constant::*)(double))
                       &dolfin::Constant::operator=)
      */
      .def("str", &dolfin::Constant::str);

    // dolfin::FacetArea
    py::class_<dolfin::FacetArea, std::shared_ptr<dolfin::FacetArea>,
               dolfin::Expression, dolfin::GenericFunction>
      (m, "FacetArea")
      .def(py::init<std::shared_ptr<const dolfin::Mesh>>());

    // dolfin::MeshCoordinates
    py::class_<dolfin::MeshCoordinates, std::shared_ptr<dolfin::MeshCoordinates>,
               dolfin::Expression, dolfin::GenericFunction>
      (m, "MeshCoordinates")
      .def(py::init<std::shared_ptr<const dolfin::Mesh>>());

    // dolfin::Function
    py::class_<dolfin::Function, std::shared_ptr<dolfin::Function>, dolfin::GenericFunction>
      (m, "Function", "A finite element function")
      .def(py::init<std::shared_ptr<const dolfin::FunctionSpace>>(), "Create a function on the given function space")
      .def(py::init<dolfin::Function&, std::size_t>())
      .def(py::init<std::shared_ptr<dolfin::FunctionSpace>, std::shared_ptr<dolfin::GenericVector>>())
      .def("_assign", (const dolfin::Function& (dolfin::Function::*)(const dolfin::Function&))
           &dolfin::Function::operator=)
      .def("_assign", (const dolfin::Function& (dolfin::Function::*)(const dolfin::Expression&))
           &dolfin::Function::operator=)
      .def("_assign", (void (dolfin::Function::*)(const dolfin::FunctionAXPY&))
           &dolfin::Function::operator=)
      .def("_in", &dolfin::Function::in)
      .def("__call__", [](dolfin::Function& self, std::vector<double>& p)
          {
            // FIXME - remove Array and replace with Eigen in DOLFIN
            const dolfin::Array<double> x(p.size(), p.data());
            Eigen::VectorXd values(self.value_size());
            dolfin::Array<double> _values(self.value_size(), values.data());
            self.eval(_values, x);
            return values;
          })
      .def("extrapolate", &dolfin::Function::extrapolate)
      .def("extrapolate", [](dolfin::Function& instance, const py::object v)
           {
             auto _v = v.attr("_cpp_object").cast<dolfin::Function*>();
             instance.extrapolate(*_v);
           })
      .def("sub", &dolfin::Function::operator[])
      .def("get_allow_extrapolation", &dolfin::Function::get_allow_extrapolation)
      .def("interpolate", (void (dolfin::Function::*)(const dolfin::GenericFunction&))
           &dolfin::Function::interpolate, "Interpolate the function u")
      .def("interpolate", [](dolfin::Function& instance, const py::object v)
           {
             auto _v = v.attr("_cpp_object").cast<dolfin::GenericFunction*>();
             instance.interpolate(*_v);
           }, "Interpolate the function u")
      .def("set_allow_extrapolation", &dolfin::Function::set_allow_extrapolation)
      // FIXME: A lot of error when using non-const version - misused
      // by Python interface?
      .def("vector", (std::shared_ptr<const dolfin::GenericVector> (dolfin::Function::*)() const)
           &dolfin::Function::vector, "Return the vector associated with the finite element Function");

    // FIXME: why is this floating here?
    m.def("interpolate", [](const dolfin::GenericFunction& f,
                          std::shared_ptr<const dolfin::FunctionSpace> V)
          {
            auto g = std::make_shared<dolfin::Function>(V);
            g->interpolate(f);
            return g;
          });

    // dolfin::FunctionAXPY
    py::class_<dolfin::FunctionAXPY, std::shared_ptr<dolfin::FunctionAXPY>> function_axpy(m, "FunctionAXPY");
    function_axpy
      .def(py::init<std::shared_ptr<const dolfin::Function>, double>())
      .def(py::init<std::shared_ptr<const dolfin::Function>, std::shared_ptr<const dolfin::Function>,
           dolfin::FunctionAXPY::Direction>())
      .def(py::init<std::vector<std::pair<double, std::shared_ptr<const dolfin::Function>>>>())
      .def(py::init([](std::vector<std::pair<double, py::object>> fun)
           {
             std::vector<std::pair<double, std::shared_ptr<const dolfin::Function>>> a;
             for (auto p : fun)
               a.push_back({p.first, p.second.attr("_cpp_object").cast<std::shared_ptr<const dolfin::Function>>()});
             return dolfin::FunctionAXPY(a);
           }))
      .def(py::init([](py::object f1, double a)
           {
             auto _f1 = f1.attr("_cpp_object").cast<std::shared_ptr<const dolfin::Function>>();
             return dolfin::FunctionAXPY(_f1, a);
           }))
      .def(py::self + py::self)
      .def(py::self + std::shared_ptr<const dolfin::Function>())
      .def("__add__", [](dolfin::FunctionAXPY& self, py::object f1)
           {
             return (self + f1.attr("_cpp_object").cast<std::shared_ptr<const dolfin::Function>>());
           })
      .def(py::self - py::self)
      .def(py::self - std::shared_ptr<const dolfin::Function>())
      .def("__sub__", [](dolfin::FunctionAXPY& self, py::object f1)
           {
             return (self - f1.attr("_cpp_object").cast<std::shared_ptr<const dolfin::Function>>());
           })
      .def(py::self * float())
      .def(py::self / float());

    // dolfin::FunctionAXPY enum
    py::enum_<dolfin::FunctionAXPY::Direction>(function_axpy, "Direction")
      .value("ADD_ADD", dolfin::FunctionAXPY::Direction::ADD_ADD)
      .value("SUB_ADD", dolfin::FunctionAXPY::Direction::SUB_ADD)
      .value("ADD_SUB", dolfin::FunctionAXPY::Direction::ADD_SUB)
      .value("SUB_SUB", dolfin::FunctionAXPY::Direction::SUB_SUB);

    // dolfin::FunctionSpace
    py::class_<dolfin::FunctionSpace, std::shared_ptr<dolfin::FunctionSpace>, dolfin::Variable>
      (m, "FunctionSpace", py::dynamic_attr())
      .def(py::init<std::shared_ptr<dolfin::Mesh>, std::shared_ptr<dolfin::FiniteElement>,
           std::shared_ptr<dolfin::GenericDofMap>>())
      .def(py::init<const dolfin::FunctionSpace&>())
      .def("__eq__", &dolfin::FunctionSpace::operator==)
      .def("dim", &dolfin::FunctionSpace::dim)
      .def("collapse", [](dolfin::FunctionSpace& self)
           {
             std::unordered_map<std::size_t, std::size_t> dofs;
             auto V = self.collapse(dofs);
             return std::pair<std::shared_ptr<dolfin::FunctionSpace>, std::unordered_map<std::size_t, std::size_t>>({V, dofs});
           })
      .def("component", &dolfin::FunctionSpace::component)
      .def("contains", &dolfin::FunctionSpace::contains)
      .def("element", &dolfin::FunctionSpace::element)
      .def("mesh", &dolfin::FunctionSpace::mesh)
      .def("dofmap", &dolfin::FunctionSpace::dofmap)
      .def("set_x", &dolfin::FunctionSpace::set_x)
      .def("sub", (std::shared_ptr<dolfin::FunctionSpace> (dolfin::FunctionSpace::*)(std::size_t) const)
           &dolfin::FunctionSpace::sub)
      .def("extract_sub_space", &dolfin::FunctionSpace::extract_sub_space)
      .def("tabulate_dof_coordinates", [](const dolfin::FunctionSpace& self)
           {
             const std::size_t gdim = self.element()->geometric_dimension();
             std::vector<double> coords = self.tabulate_dof_coordinates();
             assert(coords.size() % gdim  == 0);

             py::array_t<double> c({coords.size()/gdim, gdim}, coords.data() );
             return c;
           });

    // dolfin::LagrangeInterpolator
    py::class_<dolfin::LagrangeInterpolator> (m, "LagrangeInterpolator")
      .def_static("interpolate", (void (*)(dolfin::Function&, const dolfin::Function&))
                  &dolfin::LagrangeInterpolator::interpolate)
      .def_static("interpolate", [](py::object f1, py::object f2)
                  {
                    auto _f1 = f1.attr("_cpp_object").cast<dolfin::Function*>();
                    auto _f2cpp = f2.attr("_cpp_object");
                    if (py::isinstance<dolfin::Function>(_f2cpp))
                    {
                      auto _f2 = _f2cpp.cast<const dolfin::Function*>();
                      dolfin::LagrangeInterpolator::interpolate(*_f1, *_f2);
                    }
                    else if (py::isinstance<dolfin::Expression>(_f2cpp))
                    {
                      auto _f2 = _f2cpp.cast<const dolfin::Expression*>();
                      dolfin::LagrangeInterpolator::interpolate(*_f1, *_f2);
                    }
                    else
                      throw py::type_error("Can only interpolate Expression or Function");
                  });

    // dolfin::FunctionAssigner
    py::class_<dolfin::FunctionAssigner, std::shared_ptr<dolfin::FunctionAssigner>>
      (m, "FunctionAssigner")
      .def(py::init<std::shared_ptr<const dolfin::FunctionSpace>,
           std::shared_ptr<const dolfin::FunctionSpace>>())
      .def(py::init<std::vector<std::shared_ptr<const dolfin::FunctionSpace>>,
           std::shared_ptr<const dolfin::FunctionSpace>>())
      .def(py::init<std::shared_ptr<const dolfin::FunctionSpace>,
           std::vector<std::shared_ptr<const dolfin::FunctionSpace>>>())
      .def(py::init([](py::object V0, py::object V1)
           {
             if (py::isinstance<py::list>(V0))
             {
               std::vector<std::shared_ptr<const dolfin::FunctionSpace>> _V0;
               for (auto V : py::cast<py::list>(V0))
                 _V0.push_back(V.attr("_cpp_object").cast<std::shared_ptr<const dolfin::FunctionSpace>>());
               auto _V1 = V1.attr("_cpp_object").cast<std::shared_ptr<const dolfin::FunctionSpace>>();
               return dolfin::FunctionAssigner(_V0, _V1);
             }
             else if (py::isinstance<py::list>(V1))
             {
               std::vector<std::shared_ptr<const dolfin::FunctionSpace>> _V1;
               for (auto V : py::cast<py::list>(V1))
                 _V1.push_back(V.attr("_cpp_object").cast<std::shared_ptr<const dolfin::FunctionSpace>>());
               auto _V0 = V0.attr("_cpp_object").cast<std::shared_ptr<const dolfin::FunctionSpace>>();
               return dolfin::FunctionAssigner(_V0, _V1);
             }
             else
             {
               auto _V0 = V0.attr("_cpp_object").cast<std::shared_ptr<const dolfin::FunctionSpace>>();
               auto _V1 = V1.attr("_cpp_object").cast<std::shared_ptr<const dolfin::FunctionSpace>>();
               return dolfin::FunctionAssigner(_V0, _V1);
             }
           }))
      .def("assign", (void (dolfin::FunctionAssigner::*)(std::shared_ptr<dolfin::Function>,
                                                         std::shared_ptr<const dolfin::Function>) const)
           &dolfin::FunctionAssigner::assign)
      .def("assign", [](const dolfin::FunctionAssigner& self, py::object v0, py::object v1)
           {
             if (py::isinstance<py::list>(v0))
             {
               std::vector<std::shared_ptr<dolfin::Function>> _v0;
               for (auto v : py::cast<py::list>(v0))
                 _v0.push_back(v.attr("_cpp_object").cast<std::shared_ptr<dolfin::Function>>());
               auto _v1 = v1.attr("_cpp_object").cast<std::shared_ptr<const dolfin::Function>>();
               self.assign(_v0, _v1);
               return;
             }
             else if (py::isinstance<py::list>(v1))
             {
               auto _v0 = v0.attr("_cpp_object").cast<std::shared_ptr<dolfin::Function>>();
               std::vector<std::shared_ptr<const dolfin::Function>> _v1;
               for (auto v : py::cast<py::list>(v1))
                 _v1.push_back(v.attr("_cpp_object").cast<std::shared_ptr<const dolfin::Function>>());
               self.assign(_v0, _v1);
               return;
             }
             else
             {
               auto _v0 = v0.attr("_cpp_object").cast<std::shared_ptr<dolfin::Function>>();
               auto _v1 = v1.attr("_cpp_object").cast<std::shared_ptr<const dolfin::Function>>();
               self.assign(_v0, _v1);
               return;
             }
           });

    // dolfin::assign interface
    m.def("assign", [](py::object v0, py::object v1)
           {
             if (py::isinstance<py::list>(v0))
             {
               std::vector<std::shared_ptr<dolfin::Function>> _v0;
               for (auto v : py::cast<py::list>(v0))
                 _v0.push_back(v.attr("_cpp_object").cast<std::shared_ptr<dolfin::Function>>());
               auto _v1 = v1.attr("_cpp_object").cast<std::shared_ptr<const dolfin::Function>>();
               dolfin::assign(_v0, _v1);
               return;
             }
             else if (py::isinstance<py::list>(v1))
             {
               auto _v0 = v0.attr("_cpp_object").cast<std::shared_ptr<dolfin::Function>>();
               std::vector<std::shared_ptr<const dolfin::Function>> _v1;
               for (auto v : py::cast<py::list>(v1))
                 _v1.push_back(v.attr("_cpp_object").cast<std::shared_ptr<const dolfin::Function>>());
               dolfin::assign(_v0, _v1);
               return;
             }
             else
             {
               auto _v0 = v0.attr("_cpp_object").cast<std::shared_ptr<dolfin::Function>>();
               auto _v1 = v1.attr("_cpp_object").cast<std::shared_ptr<const dolfin::Function>>();
               dolfin::assign(_v0, _v1);
               return;
             }
           });
  }
}
