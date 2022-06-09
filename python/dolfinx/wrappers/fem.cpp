// Copyright (C) 2017-2021 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "array.h"
#include "caster_mpi.h"
#include "caster_petsc.h"
#include <array>
#include <cstdint>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/fem/CoordinateElement.h>
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/ElementDofLayout.h>
#include <dolfinx/fem/Expression.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/fem/assembler.h>
#include <dolfinx/fem/discreteoperators.h>
#include <dolfinx/fem/dofmapbuilder.h>
#include <dolfinx/fem/interpolate.h>
#include <dolfinx/fem/petsc.h>
#include <dolfinx/fem/sparsitybuild.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/geometry/BoundingBoxTree.h>
#include <dolfinx/graph/ordering.h>
#include <dolfinx/la/MatrixCSR.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/la/petsc.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/MeshTags.h>
#include <memory>
#include <petsc4py/petsc4py.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <string>
#include <ufcx.h>
#include <utility>
#include <xtensor/xadapt.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

namespace py = pybind11;

namespace
{
template <typename T>
std::map<std::pair<dolfinx::fem::IntegralType, int>,
         std::pair<xtl::span<const T>, int>>
py_to_cpp_coeffs(const std::map<std::pair<dolfinx::fem::IntegralType, int>,
                                py::array_t<T, py::array::c_style>>& coeffs)
{
  using Key_t = typename std::remove_reference_t<decltype(coeffs)>::key_type;
  std::map<Key_t, std::pair<xtl::span<const T>, int>> c;
  std::transform(coeffs.cbegin(), coeffs.cend(), std::inserter(c, c.end()),
                 [](auto& e) -> typename decltype(c)::value_type {
                   return {e.first, {e.second, e.second.shape(1)}};
                 });
  return c;
}

// Declare assembler function that have multiple scalar types
template <typename T>
void declare_functions(py::module& m)
{
  // Coefficient/constant packing
  m.def(
      "pack_coefficients",
      [](const dolfinx::fem::Form<T>& form)
      {
        using Key_t = typename std::pair<dolfinx::fem::IntegralType, int>;

        // Pack coefficients
        std::map<Key_t, std::pair<std::vector<T>, int>> coeffs
            = dolfinx::fem::allocate_coefficient_storage(form);
        dolfinx::fem::pack_coefficients(form, coeffs);

        // Move into NumPy data structures
        std::map<Key_t, py::array_t<T, py::array::c_style>> c;
        std::transform(
            coeffs.begin(), coeffs.end(), std::inserter(c, c.end()),
            [](auto& e) -> typename decltype(c)::value_type
            {
              int num_ents = e.second.first.empty()
                                 ? 0
                                 : e.second.first.size() / e.second.second;
              return {e.first, dolfinx_wrappers::as_pyarray(
                                   std::move(e.second.first),
                                   std::array{num_ents, e.second.second})};
            });

        return c;
      },
      "Pack coefficients for a Form.");
  m.def(
      "pack_constants",
      [](const dolfinx::fem::Form<T>& form) {
        return dolfinx_wrappers::as_pyarray(dolfinx::fem::pack_constants(form));
      },
      "Pack constants for a Form.");
  m.def(
      "pack_constants",
      [](const dolfinx::fem::Expression<T>& e)
      { return dolfinx_wrappers::as_pyarray(dolfinx::fem::pack_constants(e)); },
      "Pack constants for an Expression.");

  // Functional
  m.def(
      "assemble_scalar",
      [](const dolfinx::fem::Form<T>& M,
         const py::array_t<T, py::array::c_style>& constants,
         const std::map<std::pair<dolfinx::fem::IntegralType, int>,
                        py::array_t<T, py::array::c_style>>& coefficients)
      {
        return dolfinx::fem::assemble_scalar<T>(M, constants,
                                                py_to_cpp_coeffs(coefficients));
      },
      py::arg("M"), py::arg("constants"), py::arg("coefficients"),
      "Assemble functional over mesh with provided constants and "
      "coefficients");
  // Vector
  m.def(
      "assemble_vector",
      [](py::array_t<T, py::array::c_style> b, const dolfinx::fem::Form<T>& L,
         const py::array_t<T, py::array::c_style>& constants,
         const std::map<std::pair<dolfinx::fem::IntegralType, int>,
                        py::array_t<T, py::array::c_style>>& coefficients)
      {
        dolfinx::fem::assemble_vector<T>(xtl::span(b.mutable_data(), b.size()),
                                         L, constants,
                                         py_to_cpp_coeffs(coefficients));
      },
      py::arg("b"), py::arg("L"), py::arg("constants"), py::arg("coeffs"),
      "Assemble linear form into an existing vector with pre-packed constants "
      "and coefficients");
  // MatrixCSR
  m.def(
      "assemble_matrix",
      [](dolfinx::la::MatrixCSR<T>& A, const dolfinx::fem::Form<T>& a,
         const py::array_t<T, py::array::c_style>& constants,
         const std::map<std::pair<dolfinx::fem::IntegralType, int>,
                        py::array_t<T, py::array::c_style>>& coefficients,
         const std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<T>>>&
             bcs)
      {
        if (a.function_spaces()[0]->dofmap()->bs() != 1
            or a.function_spaces()[0]->dofmap()->bs() != 1)
        {
          throw std::runtime_error("Assembly with block size > 1 not yet "
                                   "supported with la::MatrixCSR.");
        }
        dolfinx::fem::assemble_matrix(A.mat_add_values(), a,
                                      xtl::span(constants),
                                      py_to_cpp_coeffs(coefficients), bcs);
      },
      "Experimental.");
  m.def(
      "insert_diagonal",
      [](dolfinx::la::MatrixCSR<T>& A, const dolfinx::fem::FunctionSpace& V,
         const std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<T>>>&
             bcs,
         T diagonal)
      { dolfinx::fem::set_diagonal(A.mat_set_values(), V, bcs, diagonal); });
  m.def(
      "assemble_matrix",
      [](const std::function<int(const py::array_t<std::int32_t>&,
                                 const py::array_t<std::int32_t>&,
                                 const py::array_t<T>&)>& fin,
         const dolfinx::fem::Form<T>& form,
         const std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<T>>>&
             bcs)
      {
        auto f = [&fin](const xtl::span<const std::int32_t>& rows,
                        const xtl::span<const std::int32_t>& cols,
                        const xtl::span<const T>& data)
        {
          return fin(py::array(rows.size(), rows.data()),
                     py::array(cols.size(), cols.data()),
                     py::array(data.size(), data.data()));
        };
        dolfinx::fem::assemble_matrix(f, form, bcs);
      },
      "Experimental assembly with Python insertion function. This will be "
      "slow. Use for testing only.");

  // BC modifiers
  m.def(
      "apply_lifting",
      [](py::array_t<T, py::array::c_style> b,
         const std::vector<std::shared_ptr<const dolfinx::fem::Form<T>>>& a,
         const std::vector<py::array_t<T, py::array::c_style>>& constants,
         const std::vector<std::map<std::pair<dolfinx::fem::IntegralType, int>,
                                    py::array_t<T, py::array::c_style>>>&
             coeffs,
         const std::vector<std::vector<
             std::shared_ptr<const dolfinx::fem::DirichletBC<T>>>>& bcs1,
         const std::vector<py::array_t<T, py::array::c_style>>& x0,
         double scale)
      {
        std::vector<xtl::span<const T>> _x0;
        for (const auto& x : x0)
          _x0.emplace_back(x.data(), x.size());

        std::vector<xtl::span<const T>> _constants;
        std::transform(constants.cbegin(), constants.cend(),
                       std::back_inserter(_constants),
                       [](auto& c) { return c; });

        std::vector<std::map<std::pair<dolfinx::fem::IntegralType, int>,
                             std::pair<xtl::span<const T>, int>>>
            _coeffs;
        std::transform(coeffs.cbegin(), coeffs.cend(),
                       std::back_inserter(_coeffs),
                       [](auto& c) { return py_to_cpp_coeffs(c); });

        dolfinx::fem::apply_lifting<T>(xtl::span(b.mutable_data(), b.size()), a,
                                       _constants, _coeffs, bcs1, _x0, scale);
      },
      "Modify vector for lifted boundary conditions");
  m.def(
      "set_bc",
      [](py::array_t<T, py::array::c_style> b,
         const std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<T>>>&
             bcs,
         const py::array_t<T, py::array::c_style>& x0, double scale)
      {
        if (x0.ndim() == 0)
        {
          dolfinx::fem::set_bc<T>(xtl::span(b.mutable_data(), b.size()), bcs,
                                  scale);
        }
        else if (x0.ndim() == 1)
        {
          dolfinx::fem::set_bc<T>(xtl::span(b.mutable_data(), b.size()), bcs,
                                  xtl::span(x0.data(), x0.shape(0)), scale);
        }
        else
          throw std::runtime_error("Wrong array dimension.");
      },
      py::arg("b"), py::arg("bcs"), py::arg("x0") = py::none(),
      py::arg("scale") = 1.0);
}

// Declare DirichletBC objects for type T
template <typename T>
void declare_objects(py::module& m, const std::string& type)
{
  // dolfinx::fem::DirichletBC
  std::string pyclass_name = std::string("DirichletBC_") + type;
  py::class_<dolfinx::fem::DirichletBC<T>,
             std::shared_ptr<dolfinx::fem::DirichletBC<T>>>
      dirichletbc(m, pyclass_name.c_str(),
                  "Object for representing Dirichlet (essential) boundary "
                  "conditions");

  dirichletbc
      .def(
          py::init(
              [](T g, const py::array_t<std::int32_t, py::array::c_style>& dofs,
                 const std::shared_ptr<const dolfinx::fem::FunctionSpace>& V)
              {
                return dolfinx::fem::DirichletBC<T>(
                    g, std::vector(dofs.data(), dofs.data() + dofs.size()), V);
              }),
          py::arg("g"), py::arg("dofs").noconvert(), py::arg("V"))
      .def(
          py::init(
              [](const py::array_t<T>& g,
                 const py::array_t<std::int32_t, py::array::c_style>& dofs,
                 const std::shared_ptr<const dolfinx::fem::FunctionSpace>& V)
              {
                if (dofs.ndim() != 1)
                  throw std::runtime_error("Wrong number of dims");
                std::vector<std::size_t> strides;
                std::transform(g.strides(), g.strides() + g.ndim(),
                               std::back_inserter(strides),
                               [](auto s) { return s / sizeof(T); });
                std::vector<std::size_t> shape(g.shape(), g.shape() +
                g.ndim()); auto _g = xt::adapt(g.data(), g.size(),
                xt::no_ownership(),
                                    shape, strides);
                return dolfinx::fem::DirichletBC<T>(
                    _g, std::vector(dofs.data(), dofs.data() + dofs.size()),
                    V);
              }),
          py::arg("g").noconvert(), py::arg("dofs").noconvert(),
          py::arg("V"))
      .def(py::init(
               [](const std::shared_ptr<const dolfinx::fem::Constant<T>>& g,
                  const py::array_t<std::int32_t, py::array::c_style>& dofs,
                  const std::shared_ptr<const dolfinx::fem::FunctionSpace>&
                  V)
               {
                 return dolfinx::fem::DirichletBC<T>(
                     g, std::vector(dofs.data(), dofs.data() + dofs.size()),
                     V);
               }),
           py::arg("g").noconvert(), py::arg("dofs").noconvert(),
           py::arg("V"))
      .def(py::init(
               [](const std::shared_ptr<const dolfinx::fem::Function<T>>& g,
                  const py::array_t<std::int32_t, py::array::c_style>& dofs)
               {
                 return dolfinx::fem::DirichletBC<T>(
                     g, std::vector(dofs.data(), dofs.data() + dofs.size()));
               }),
           py::arg("g").noconvert(), py::arg("dofs"))
      .def(
          py::init(
              [](const std::shared_ptr<const dolfinx::fem::Function<T>>& g,
                 const std::array<py::array_t<std::int32_t,
                 py::array::c_style>,
                                  2>& V_g_dofs,
                 const std::shared_ptr<const dolfinx::fem::FunctionSpace>& V)
              {
                std::array dofs = {std::vector(V_g_dofs[0].data(),
                V_g_dofs[0].data() + V_g_dofs[0].size()),
                                   std::vector(V_g_dofs[1].data(),
                                   V_g_dofs[1].data() + V_g_dofs[1].size())};
                return dolfinx::fem::DirichletBC(g, std::move(dofs), V);
              }),
          py::arg("g").noconvert(), py::arg("dofs").noconvert(),
          py::arg("V").noconvert())
      .def_property_readonly("dtype", [](const dolfinx::fem::Form<T>& self)
                             { return py::dtype::of<T>(); })
      .def("dof_indices",
           [](const dolfinx::fem::DirichletBC<T>& self)
           {
             auto [dofs, owned] = self.dof_indices();
             return std::pair(py::array_t<std::int32_t>(
                                  dofs.size(), dofs.data(), py::cast(self)),
                              owned);
           })
      .def_property_readonly("function_space",
                             &dolfinx::fem::DirichletBC<T>::function_space)
      .def_property_readonly("value", &dolfinx::fem::DirichletBC<T>::value);

  // dolfinx::fem::Function
  std::string pyclass_name_function = std::string("Function_") + type;
  py::class_<dolfinx::fem::Function<T>,
             std::shared_ptr<dolfinx::fem::Function<T>>>(
      m, pyclass_name_function.c_str(), "A finite element function")
      .def(py::init<std::shared_ptr<const dolfinx::fem::FunctionSpace>>(),
           "Create a function on the given function space")
      .def(py::init<std::shared_ptr<dolfinx::fem::FunctionSpace>,
                    std::shared_ptr<dolfinx::la::Vector<T>>>())
      .def_readwrite("name", &dolfinx::fem::Function<T>::name)
      .def("sub", &dolfinx::fem::Function<T>::sub,
           "Return sub-function (view into parent Function")
      .def("collapse", &dolfinx::fem::Function<T>::collapse,
           "Collapse sub-function view")
      .def(
          "interpolate",
          [](dolfinx::fem::Function<T>& self,
             const std::function<py::array_t<T>(const py::array_t<double>&)>& f,
             const py::array_t<std::int32_t, py::array::c_style>& cells)
          {
            auto _f = [&f](const xt::xtensor<double, 2>& x) -> xt::xarray<T>
            {
              auto strides = x.strides();
              std::transform(strides.begin(), strides.end(), strides.begin(),
                             [](auto s) { return s * sizeof(double); });
              py::array_t _x(x.shape(), strides, x.data(), py::none());
              py::array_t v = f(_x);
              std::vector<std::size_t> shape;
              std::copy_n(v.shape(), v.ndim(), std::back_inserter(shape));
              return xt::adapt(v.data(), shape);
            };
            self.interpolate(_f, cells);
          },
          py::arg("f"), py::arg("cells"), "Interpolate an expression function")
      .def(
          "interpolate",
          [](dolfinx::fem::Function<T>& self, dolfinx::fem::Function<T>& u,
             const py::array_t<std::int32_t, py::array::c_style>& cells)
          { self.interpolate(u, cells); },
          py::arg("u"), py::arg("cells"),
          "Interpolate a finite element function")
      .def(
          "interpolate_ptr",
          [](dolfinx::fem::Function<T>& self, std::uintptr_t addr,
             const py::array_t<std::int32_t, py::array::c_style>& cells)
          {
            assert(self.function_space());
            auto element = self.function_space()->element();
            assert(element);

            // Compute value size
            auto vshape = element->value_shape();
            std::size_t value_size = std::reduce(
                std::begin(vshape), std::end(vshape), 1, std::multiplies<>());

            std::function<void(T*, int, int, const double*)> f
                = reinterpret_cast<void (*)(T*, int, int, const double*)>(addr);
            auto _f = [&f, value_size](const xt::xtensor<double, 2>& x)
            {
              xt::xarray<T> values = xt::empty<T>({value_size, x.shape(1)});
              f(values.data(), values.shape(1), values.shape(0), x.data());
              return values;
            };

            self.interpolate(_f, cells);
          },
          py::arg("f"), py::arg("cells"),
          "Interpolate using a pointer to an Expression with a C signature")
      .def(
          "interpolate",
          [](dolfinx::fem::Function<T>& self,
             const dolfinx::fem::Expression<T>& expr,
             const py::array_t<std::int32_t, py::array::c_style>& cells)
          { self.interpolate(expr, cells); },
          "Interpolate an Expression on a set of cells")
      .def_property_readonly(
          "x", py::overload_cast<>(&dolfinx::fem::Function<T>::x),
          "Return the vector associated with the finite element Function")
      .def(
          "eval",
          [](const dolfinx::fem::Function<T>& self,
             const py::array_t<double, py::array::c_style>& x,
             const py::array_t<std::int32_t, py::array::c_style>& cells,
             py::array_t<T, py::array::c_style>& u)
          {
            // TODO: handle 1d case

            std::vector<std::size_t> shape_x(x.shape(), x.shape() + 2);
            auto _x
                = xt::adapt(x.data(), x.size(), xt::no_ownership(), shape_x);

            std::array<std::size_t, 2> shape_u;
            std::copy_n(u.shape(), 2, shape_u.begin());

            // The below should work, but misbehaves with the Intel
            // icpx compiler
            // xt::xtensor<T, 2> _u = xt::adapt(u.mutable_data(), u.size(),
            //                                  xt::no_ownership(), shape_u);
            xt::xtensor<T, 2> _u(shape_u);
            std::copy_n(u.data(), u.size(), _u.data());

            self.eval(_x, xtl::span(cells.data(), cells.size()), _u);
            std::copy_n(_u.data(), _u.size(), u.mutable_data());
          },
          py::arg("x"), py::arg("cells"), py::arg("values"),
          "Evaluate Function")
      .def_property_readonly("function_space",
                             &dolfinx::fem::Function<T>::function_space);

  // dolfinx::fem::Constant
  std::string pyclass_name_constant = std::string("Constant_") + type;
  py::class_<dolfinx::fem::Constant<T>,
             std::shared_ptr<dolfinx::fem::Constant<T>>>(
      m, pyclass_name_constant.c_str(),
      "Value constant with respect to integration domain")
      .def(py::init([](T c) { return dolfinx::fem::Constant<T>(c); }),
           py::arg("c").noconvert(), "Create a constant from a scalar")
      .def(py::init(
               [](const py::array_t<T, py::array::c_style>& c)
               {
                 std::vector<std::size_t> s;
                 std::copy_n(c.shape(), c.ndim(), std::back_inserter(s));
                 return dolfinx::fem::Constant<T>(
                     xt::adapt(c.data(), c.size(), xt::no_ownership(), s));
               }),
           py::arg("c").noconvert(), "Create a constant from a value array")
      .def_property_readonly("dtype", [](const dolfinx::fem::Constant<T>& self)
                             { return py::dtype::of<T>(); })
      .def_property_readonly(
          "value",
          [](dolfinx::fem::Constant<T>& self)
          { return py::array(self.shape, self.value.data(), py::none()); },
          py::return_value_policy::reference_internal);

  // dolfinx::fem::Expression
  std::string pyclass_name_expr = std::string("Expression_") + type;
  py::
      class_<dolfinx::fem::Expression<T>,
             std::shared_ptr<dolfinx::fem::Expression<T>>>(
          m, pyclass_name_expr.c_str(), "An Expression")
          .def(
              py::init(
                  [](const std::vector<std::shared_ptr<
                         const dolfinx::fem::Function<T>>>& coefficients,
                     const std::vector<std::shared_ptr<
                         const dolfinx::fem::Constant<T>>>& constants,
                     const py::array_t<double, py::array::c_style>& X,
                     std::uintptr_t fn_addr,
                     const std::vector<int>& value_shape,
                     const std::shared_ptr<const dolfinx::mesh::Mesh>& mesh,
                     const std::shared_ptr<const dolfinx::fem::FunctionSpace>&
                         argument_function_space)
                  {
                    auto tabulate_expression_ptr
                        = (void (*)(T*, const T*, const T*, const double*,
                                    const int*, const std::uint8_t*))fn_addr;
                    auto _x_ref = xt::adapt(X.data(), {X.shape(0), X.shape(1)});
                    return dolfinx::fem::Expression<T>(
                        coefficients, constants, _x_ref,
                        tabulate_expression_ptr, value_shape, mesh,
                        argument_function_space);
                  }),
              py::arg("coefficients"), py::arg("constants"), py::arg("X"),
              py::arg("fn"), py::arg("value_shape"), py::arg("mesh"),
              py::arg("argument_function_space"))
          .def(
              "eval",
              [](const dolfinx::fem::Expression<T>& self,
                 const py::
                     array_t<std::int32_t, py::array::c_style>& active_cells,
                 py::array_t<T, py::array::c_style>& values)
              {
                const int size = values.shape(0) * values.shape(1);
                auto _values = xt::adapt(
                    const_cast<T*>(values.data()), size, xt::no_ownership(),
                    std::array<std::size_t, 2>({(std::size_t)values.shape(0),
                                                (std::size_t)values.shape(1)}));
                self.eval(xtl::span(active_cells.data(), active_cells.size()),
                          _values);
              })
          .def_property_readonly("dtype",
                                 [](const dolfinx::fem::Expression<T>& self)
                                 { return py::dtype::of<T>(); })
          .def_property_readonly("mesh", &dolfinx::fem::Expression<T>::mesh)
          .def_property_readonly("value_size",
                                 &dolfinx::fem::Expression<T>::value_size)
          .def_property_readonly("value_shape",
                                 &dolfinx::
                                     fem::Expression<T>::value_shape)
          .def_property_readonly("X",
                                 [](const dolfinx::fem::Expression<T>& self)
                                 {
                                   return py::array_t<double>(self.X().shape(),
                                                              self.X().data(),
                                                              py::cast(self));
                                 });

  std::string pymethod_create_expression
      = std::string("create_expression_") + type;
  m.def(
      pymethod_create_expression.c_str(),
      [](const std::uintptr_t expression,
         const std::vector<std::shared_ptr<const dolfinx::fem::Function<T>>>&
             coefficients,
         const std::vector<std::shared_ptr<const dolfinx::fem::Constant<T>>>&
             constants,
         const std::shared_ptr<const dolfinx::mesh::Mesh>& mesh,
         const std::shared_ptr<const dolfinx::fem::FunctionSpace>&
             argument_function_space)
      {
        const ufcx_expression* p
            = reinterpret_cast<const ufcx_expression*>(expression);
        return dolfinx::fem::create_expression<T>(
            *p, coefficients, constants, mesh, argument_function_space);
      },
      "Create Form from a pointer to ufc_form.");
}

void petsc_module(py::module& m)
{
  // Create PETSc vectors and matrices
  m.def("create_vector_block", &dolfinx::fem::petsc::create_vector_block,
        py::return_value_policy::take_ownership,
        "Create a monolithic vector for multiple (stacked) linear forms.");
  m.def("create_vector_nest", &dolfinx::fem::petsc::create_vector_nest,
        py::return_value_policy::take_ownership,
        "Create nested vector for multiple (stacked) linear forms.");
  m.def("create_matrix", dolfinx::fem::petsc::create_matrix,
        py::return_value_policy::take_ownership, py::arg("a"),
        py::arg("type") = std::string(),
        "Create a PETSc Mat for bilinear form.");
  m.def("create_matrix_block", &dolfinx::fem::petsc::create_matrix_block,
        py::return_value_policy::take_ownership, py::arg("a"),
        py::arg("type") = std::string(),
        "Create monolithic sparse matrix for stacked bilinear forms.");
  m.def("create_matrix_nest", &dolfinx::fem::petsc::create_matrix_nest,
        py::return_value_policy::take_ownership, py::arg("a"),
        py::arg("types") = std::vector<std::vector<std::string>>(),
        "Create nested sparse matrix for bilinear forms.");

  // PETSc Matrices
  m.def(
      "assemble_matrix",
      [](Mat A, const dolfinx::fem::Form<PetscScalar>& a,
         const py::array_t<PetscScalar, py::array::c_style>& constants,
         const std::map<std::pair<dolfinx::fem::IntegralType, int>,
                        py::array_t<PetscScalar, py::array::c_style>>&
             coefficients,
         const std::vector<std::shared_ptr<
             const dolfinx::fem::DirichletBC<PetscScalar>>>& bcs,
         bool unrolled)
      {
        if (unrolled)
        {
          auto set_fn = dolfinx::la::petsc::Matrix::set_block_expand_fn(
              A, a.function_spaces()[0]->dofmap()->bs(),
              a.function_spaces()[1]->dofmap()->bs(), ADD_VALUES);
          dolfinx::fem::assemble_matrix(set_fn, a, xtl::span(constants),
                                        py_to_cpp_coeffs(coefficients), bcs);
        }
        else
        {
          dolfinx::fem::assemble_matrix(
              dolfinx::la::petsc::Matrix::set_block_fn(A, ADD_VALUES), a,
              xtl::span(constants), py_to_cpp_coeffs(coefficients), bcs);
        }
      },
      py::arg("A"), py::arg("a"), py::arg("constants"), py::arg("coeffs"),
      py::arg("bcs"), py::arg("unrolled") = false,
      "Assemble bilinear form into an existing PETSc matrix");
  m.def(
      "assemble_matrix",
      [](Mat A, const dolfinx::fem::Form<PetscScalar>& a,
         const py::array_t<PetscScalar, py::array::c_style>& constants,
         const std::map<std::pair<dolfinx::fem::IntegralType, int>,
                        py::array_t<PetscScalar, py::array::c_style>>&
             coefficients,
         const py::array_t<std::int8_t, py::array::c_style>& rows0,
         const py::array_t<std::int8_t, py::array::c_style>& rows1,
         bool unrolled)
      {
        if (rows0.ndim() != 1 or rows1.ndim())
        {
          throw std::runtime_error(
              "Expected 1D arrays for boundary condition rows/columns");
        }

        std::function<int(const xtl::span<const std::int32_t>&,
                          const xtl::span<const std::int32_t>&,
                          const xtl::span<const PetscScalar>&)>
            set_fn;
        if (unrolled)
        {
          set_fn = dolfinx::la::petsc::Matrix::set_block_expand_fn(
              A, a.function_spaces()[0]->dofmap()->bs(),
              a.function_spaces()[1]->dofmap()->bs(), ADD_VALUES);
        }
        else
          set_fn = dolfinx::la::petsc::Matrix::set_block_fn(A, ADD_VALUES);

        dolfinx::fem::assemble_matrix(set_fn, a, xtl::span(constants),
                                      py_to_cpp_coeffs(coefficients), rows0,
                                      rows1);
      },
      py::arg("A"), py::arg("a"), py::arg("constants"), py::arg("coeffs"),
      py::arg("rows0"), py::arg("rows1"), py::arg("unrolled") = false);
  m.def("insert_diagonal",
        [](Mat A, const dolfinx::fem::FunctionSpace& V,
           const std::vector<std::shared_ptr<
               const dolfinx::fem::DirichletBC<PetscScalar>>>& bcs,
           PetscScalar diagonal)
        {
          dolfinx::fem::set_diagonal(
              dolfinx::la::petsc::Matrix::set_fn(A, INSERT_VALUES), V, bcs,
              diagonal);
        });

  m.def(
      "discrete_gradient",
      [](const dolfinx::fem::FunctionSpace& V0,
         const dolfinx::fem::FunctionSpace& V1)
      {
        assert(V0.mesh());
        std::shared_ptr<const dolfinx::mesh::Mesh> mesh = V0.mesh();
        assert(V1.mesh());
        assert(mesh == V1.mesh());
        MPI_Comm comm = mesh->comm();

        std::shared_ptr<const dolfinx::fem::DofMap> dofmap0 = V0.dofmap();
        assert(dofmap0);
        std::shared_ptr<const dolfinx::fem::DofMap> dofmap1 = V1.dofmap();
        assert(dofmap1);

        // Create and build  sparsity pattern
        assert(dofmap0->index_map);
        assert(dofmap1->index_map);
        dolfinx::la::SparsityPattern sp(
            comm, {dofmap1->index_map, dofmap0->index_map},
            {dofmap1->index_map_bs(), dofmap0->index_map_bs()});
        dolfinx::fem::sparsitybuild::cells(sp, mesh->topology(),
                                           {*dofmap1, *dofmap0});
        sp.assemble();

        // Build operator
        Mat A = dolfinx::la::petsc::create_matrix(comm, sp);
        MatSetOption(A, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE);
        dolfinx::fem::discrete_gradient<PetscScalar>(
            V0, V1, dolfinx::la::petsc::Matrix::set_fn(A, INSERT_VALUES));
        return A;
      },
      py::return_value_policy::take_ownership);
  m.def(
      "interpolation_matrix",
      [](const dolfinx::fem::FunctionSpace& V0,
         const dolfinx::fem::FunctionSpace& V1)
      {
        assert(V0.mesh());
        std::shared_ptr<const dolfinx::mesh::Mesh> mesh = V0.mesh();
        assert(V1.mesh());
        assert(mesh == V1.mesh());
        MPI_Comm comm = mesh->comm();

        std::shared_ptr<const dolfinx::fem::DofMap> dofmap0 = V0.dofmap();
        assert(dofmap0);
        std::shared_ptr<const dolfinx::fem::DofMap> dofmap1 = V1.dofmap();
        assert(dofmap1);

        // Create and build  sparsity pattern
        assert(dofmap0->index_map);
        assert(dofmap1->index_map);
        dolfinx::la::SparsityPattern sp(
            comm, {dofmap1->index_map, dofmap0->index_map},
            {dofmap1->index_map_bs(), dofmap0->index_map_bs()});
        dolfinx::fem::sparsitybuild::cells(sp, mesh->topology(),
                                           {*dofmap1, *dofmap0});
        sp.assemble();

        // Build operator
        Mat A = dolfinx::la::petsc::create_matrix(comm, sp);
        MatSetOption(A, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE);
        dolfinx::fem::interpolation_matrix<PetscScalar>(
            V0, V1, dolfinx::la::petsc::Matrix::set_block_fn(A, INSERT_VALUES));
        return A;
      },
      py::return_value_policy::take_ownership);
}

template <typename T>
void declare_form(py::module& m, const std::string& type)
{
  // dolfinx::fem::Form
  std::string pyclass_name_form = std::string("Form_") + type;
  py::class_<dolfinx::fem::Form<T>, std::shared_ptr<dolfinx::fem::Form<T>>>(
      m, pyclass_name_form.c_str(), "Variational form object")
      .def(
          py::init(
              [](const std::vector<std::shared_ptr<
                     const dolfinx::fem::FunctionSpace>>& spaces,
                 const std::map<
                     dolfinx::fem::IntegralType,
                     std::pair<std::vector<std::pair<int, py::object>>,
                               const dolfinx::mesh::MeshTags<int>*>>& integrals,
                 const std::vector<std::shared_ptr<
                     const dolfinx::fem::Function<T>>>& coefficients,
                 const std::vector<std::shared_ptr<
                     const dolfinx::fem::Constant<T>>>& constants,
                 bool needs_permutation_data,
                 const std::shared_ptr<const dolfinx::mesh::Mesh>& mesh)
              {
                using kern
                    = std::function<void(T*, const T*, const T*, const double*,
                                         const int*, const std::uint8_t*)>;
                std::map<dolfinx::fem::IntegralType,
                         std::pair<std::vector<std::pair<int, kern>>,
                                   const dolfinx::mesh::MeshTags<int>*>>
                    _integrals;

                // Loop over kernel for each entity type
                for (auto& kernel_type : integrals)
                {
                  // Set subdomain markers
                  _integrals[kernel_type.first].second = nullptr;

                  // Loop over each domain kernel
                  for (auto& kernel : kernel_type.second.first)
                  {
                    auto tabulate_tensor_ptr
                        = (void (*)(T*, const T*, const T*, const double*,
                                    const int*, const std::uint8_t*))
                              kernel.second.cast<std::uintptr_t>();
                    _integrals[kernel_type.first].first.push_back(
                        {kernel.first, tabulate_tensor_ptr});
                  }
                }
                return dolfinx::fem::Form<T>(spaces, _integrals, coefficients,
                                             constants, needs_permutation_data,
                                             mesh);
              }),
          py::arg("spaces"), py::arg("integrals"), py::arg("coefficients"),
          py::arg("constants"), py::arg("need_permutation_data"),
          py::arg("mesh") = py::none())
      .def(py::init(
               [](std::uintptr_t form,
                  const std::vector<std::shared_ptr<
                      const dolfinx::fem::FunctionSpace>>& spaces,
                  const std::vector<std::shared_ptr<
                      const dolfinx::fem::Function<T>>>& coefficients,
                  const std::vector<std::shared_ptr<
                      const dolfinx::fem::Constant<T>>>& constants,
                  const std::map<dolfinx::fem::IntegralType,
                                 const dolfinx::mesh::MeshTags<int>*>&
                      subdomains,
                  const std::shared_ptr<const dolfinx::mesh::Mesh>& mesh)
               {
                 ufcx_form* p = reinterpret_cast<ufcx_form*>(form);
                 return dolfinx::fem::create_form<T>(
                     *p, spaces, coefficients, constants, subdomains, mesh);
               }),
           "Create a Form from a pointer to a ufcx_form")
      .def_property_readonly("dtype", [](const dolfinx::fem::Form<T>& self)
                             { return py::dtype::of<T>(); })
      .def_property_readonly("coefficients",
                             &dolfinx::fem::Form<T>::coefficients)
      .def_property_readonly("rank", &dolfinx::fem::Form<T>::rank)
      .def_property_readonly("mesh", &dolfinx::fem::Form<T>::mesh)
      .def_property_readonly("function_spaces",
                             &dolfinx::fem::Form<T>::function_spaces)
      .def("integral_ids", &dolfinx::fem::Form<T>::integral_ids)
      .def_property_readonly("integral_types",
                             &dolfinx::fem::Form<T>::integral_types)
      .def_property_readonly("needs_facet_permutations",
                             &dolfinx::fem::Form<T>::needs_facet_permutations)
      .def(
          "domains",
          [](const dolfinx::fem::Form<T>& self, dolfinx::fem::IntegralType type,
             int i) -> py::array_t<std::int32_t>
          {
            switch (type)
            {
            case dolfinx::fem::IntegralType::cell:
            {
              return py::array_t<std::int32_t>(self.cell_domains(i).size(),
                                               self.cell_domains(i).data(),
                                               py::cast(self));
            }
            case dolfinx::fem::IntegralType::exterior_facet:
            {
              const std::vector<std::pair<std::int32_t, int>>& _d
                  = self.exterior_facet_domains(i);
              std::array<py::ssize_t, 2> shape = {py::ssize_t(_d.size()), 2};
              py::array_t<std::int32_t> domains(shape);
              auto d = domains.mutable_unchecked<2>();
              for (py::ssize_t i = 0; i < d.shape(0); ++i)
              {
                d(i, 0) = _d[i].first;
                d(i, 1) = _d[i].second;
              }
              return domains;
            }
            case dolfinx::fem::IntegralType::interior_facet:
            {
              const std::vector<
                  std::tuple<std::int32_t, int, std::int32_t, int>>& _d
                  = self.interior_facet_domains(i);
              std::array<py::ssize_t, 3> shape = {py::ssize_t(_d.size()), 2, 2};
              py::array_t<std::int32_t> domains(shape);
              auto d = domains.mutable_unchecked<3>();
              for (py::ssize_t i = 0; i < d.shape(0); ++i)
              {
                d(i, 0, 0) = std::get<0>(_d[i]);
                d(i, 0, 1) = std::get<1>(_d[i]);
                d(i, 1, 0) = std::get<2>(_d[i]);
                d(i, 1, 1) = std::get<3>(_d[i]);
              }
              return domains;
            }
            default:
              throw ::std::runtime_error("Integral type unsupported.");
            }
          });

  // Form
  std::string pymethod_create_form = std::string("create_form_") + type;
  m.def(
      pymethod_create_form.c_str(),
      [](std::uintptr_t form,
         const std::vector<std::shared_ptr<const dolfinx::fem::FunctionSpace>>&
             spaces,
         const std::vector<std::shared_ptr<const dolfinx::fem::Function<T>>>&
             coefficients,
         const std::vector<std::shared_ptr<const dolfinx::fem::Constant<T>>>&
             constants,
         const std::map<dolfinx::fem::IntegralType,
                        const dolfinx::mesh::MeshTags<int>*>& subdomains,
         const std::shared_ptr<const dolfinx::mesh::Mesh>& mesh)
      {
        ufcx_form* p = reinterpret_cast<ufcx_form*>(form);
        return dolfinx::fem::create_form<T>(*p, spaces, coefficients, constants,
                                            subdomains, mesh);
      },
      "Create Form from a pointer to ufcx_form.");
}
} // namespace

namespace dolfinx_wrappers
{

void fem(py::module& m)
{
  // Load basix and dolfinx to use Pybindings
  py::module_::import("basix");

  py::module petsc_mod
      = m.def_submodule("petsc", "PETSc-specific finite element module");
  petsc_module(petsc_mod);

  // dolfinx::fem::assemble
  declare_functions<double>(m);
  declare_functions<float>(m);
  declare_functions<std::complex<double>>(m);
  // NOTE: we can't enable std::complex<float> because C++ doesn't
  // support multiplication of std::complex<float> and double.
  // declare_functions<std::complex<float>>(m);

  declare_objects<float>(m, "float32");
  declare_objects<double>(m, "float64");
  declare_objects<std::complex<float>>(m, "complex64");
  declare_objects<std::complex<double>>(m, "complex128");

  declare_form<double>(m, "float64");
  declare_form<float>(m, "float32");
  declare_form<std::complex<double>>(m, "complex128");
  declare_form<std::complex<float>>(m, "complex64");

  m.def(
      "create_sparsity_pattern",
      [](const dolfinx::mesh::Topology& topology,
         const std::vector<std::reference_wrapper<const dolfinx::fem::DofMap>>&
             dofmaps,
         const std::set<dolfinx::fem::IntegralType>& types)
      {
        if (dofmaps.size() != 2)
        {
          throw std::runtime_error(
              "create_sparsity_pattern requires exactly two dofmaps.");
        }
        return dolfinx::fem::create_sparsity_pattern(
            topology, {dofmaps[0], dofmaps[1]}, types);
      },
      "Create a sparsity pattern.");
  m.def(
      "create_element_dof_layout",
      [](std::uintptr_t dofmap, const dolfinx::mesh::CellType cell_type,
         const std::vector<int>& parent_map)
      {
        ufcx_dofmap* p = reinterpret_cast<ufcx_dofmap*>(dofmap);
        return dolfinx::fem::create_element_dof_layout(*p, cell_type,
                                                       parent_map);
      },
      "Create ElementDofLayout object from a ufc dofmap.");
  m.def(
      "create_dofmap",
      [](const MPICommWrapper comm, std::uintptr_t dofmap,
         dolfinx::mesh::Topology& topology,
         const dolfinx::fem::FiniteElement& element)
      {
        ufcx_dofmap* p = reinterpret_cast<ufcx_dofmap*>(dofmap);
        assert(p);
        dolfinx::fem::ElementDofLayout layout
            = dolfinx::fem::create_element_dof_layout(*p, topology.cell_type());
        return dolfinx::fem::create_dofmap(comm.get(), layout, topology,
                                           nullptr, element);
      },
      "Create DofMap object from a pointer to ufcx_dofmap.");
  m.def(
      "build_dofmap",
      [](const MPICommWrapper comm, const dolfinx::mesh::Topology& topology,
         const dolfinx::fem::ElementDofLayout& layout)
      {
        auto [map, bs, dofmap] = dolfinx::fem::build_dofmap_data(
            comm.get(), topology, layout,
            [](const dolfinx::graph::AdjacencyList<std::int32_t>& g)
            { return dolfinx::graph::reorder_gps(g); });
        return std::tuple(std::move(map), bs, std::move(dofmap));
      },
      "Build and dofmap on a mesh.");
  m.def("transpose_dofmap", &dolfinx::fem::transpose_dofmap,
        "Build the index to (cell, local index) map from a "
        "dofmap ((cell, local index ) -> index).");

  // dolfinx::fem::FiniteElement
  py::class_<dolfinx::fem::FiniteElement,
             std::shared_ptr<dolfinx::fem::FiniteElement>>(
      m, "FiniteElement", "Finite element object")
      .def(py::init(
          [](std::uintptr_t ufcx_element)
          {
            ufcx_finite_element* p
                = reinterpret_cast<ufcx_finite_element*>(ufcx_element);
            return dolfinx::fem::FiniteElement(*p);
          }))
      .def("__eq__", &dolfinx::fem::FiniteElement::operator==)
      .def_property_readonly("basix_element",
                             &dolfinx::fem::FiniteElement::basix_element,
                             py::return_value_policy::reference_internal)
      .def_property_readonly("num_sub_elements",
                             &dolfinx::fem::FiniteElement::num_sub_elements)
      .def_property_readonly(
          "interpolation_points",
          [](const dolfinx::fem::FiniteElement& self)
          {
            const xt::xtensor<double, 2>& x = self.interpolation_points();
            return py::array_t<double>(x.shape(), x.data(), py::cast(self));
          })
      .def_property_readonly("interpolation_ident",
                             &dolfinx::fem::FiniteElement::interpolation_ident)
      .def_property_readonly("space_dimension",
                             &dolfinx::fem::FiniteElement::space_dimension)
      .def_property_readonly("value_shape",
                             [](const dolfinx::fem::FiniteElement& self)
                             {
                               xtl::span<const int> shape = self.value_shape();
                               return py::array_t(shape.size(), shape.data(),
                                                  py::none());
                             })
      .def("apply_dof_transformation",
           [](const dolfinx::fem::FiniteElement& self,
              py::array_t<double, py::array::c_style>& x,
              std::uint32_t cell_permutation, int dim)
           {
             self.apply_dof_transformation(
                 xtl::span(x.mutable_data(), x.size()), cell_permutation, dim);
           })
      .def_property_readonly(
          "needs_dof_transformations",
          &dolfinx::fem::FiniteElement::needs_dof_transformations)
      .def("signature", &dolfinx::fem::FiniteElement::signature);

  // dolfinx::fem::ElementDofLayout
  py::class_<dolfinx::fem::ElementDofLayout,
             std::shared_ptr<dolfinx::fem::ElementDofLayout>>(
      m, "ElementDofLayout", "Object describing the layout of dofs on a cell")
      .def(py::init<int, const std::vector<std::vector<std::vector<int>>>&,
                    const std::vector<std::vector<std::vector<int>>>&,
                    const std::vector<int>&,
                    const std::vector<dolfinx::fem::ElementDofLayout>&>())
      .def_property_readonly("num_dofs",
                             &dolfinx::fem::ElementDofLayout::num_dofs)
      .def("num_entity_dofs", &dolfinx::fem::ElementDofLayout::num_entity_dofs)
      .def("num_entity_closure_dofs",
           &dolfinx::fem::ElementDofLayout::num_entity_closure_dofs)
      .def("entity_dofs", &dolfinx::fem::ElementDofLayout::entity_dofs)
      .def("entity_closure_dofs",
           &dolfinx::fem::ElementDofLayout::entity_closure_dofs)
      .def_property_readonly("block_size",
                             &dolfinx::fem::ElementDofLayout::block_size);

  // dolfinx::fem::DofMap
  py::class_<dolfinx::fem::DofMap, std::shared_ptr<dolfinx::fem::DofMap>>(
      m, "DofMap", "DofMap object")
      .def(py::init<const dolfinx::fem::ElementDofLayout&,
                    std::shared_ptr<const dolfinx::common::IndexMap>, int,
                    dolfinx::graph::AdjacencyList<std::int32_t>&, int>(),
           py::arg("element_dof_layout"), py::arg("index_map"),
           py::arg("index_map_bs"), py::arg("dofmap"), py::arg("bs"))
      .def_readonly("index_map", &dolfinx::fem::DofMap::index_map)
      .def_property_readonly("index_map_bs",
                             &dolfinx::fem::DofMap::index_map_bs)
      .def_property_readonly("dof_layout",
                             &dolfinx::fem::DofMap::element_dof_layout)
      .def("cell_dofs",
           [](const dolfinx::fem::DofMap& self, int cell)
           {
             xtl::span<const std::int32_t> dofs = self.cell_dofs(cell);
             return py::array_t<std::int32_t>(dofs.size(), dofs.data(),
                                              py::cast(self));
           })
      .def_property_readonly("bs", &dolfinx::fem::DofMap::bs)
      .def("list", &dolfinx::fem::DofMap::list,
           py::return_value_policy::reference_internal);

  // dolfinx::fem::CoordinateElement
  py::class_<dolfinx::fem::CoordinateElement,
             std::shared_ptr<dolfinx::fem::CoordinateElement>>(
      m, "CoordinateElement", "Coordinate map element")
      .def(py::init<dolfinx::mesh::CellType, int>(), py::arg("celltype"),
           py::arg("degree"))
      .def("create_dof_layout",
           &dolfinx::fem::CoordinateElement::create_dof_layout)
      .def_property_readonly("degree", &dolfinx::fem::CoordinateElement::degree)
      .def("push_forward",
           [](const dolfinx::fem::CoordinateElement& self,
              const py::array_t<double, py::array::c_style>& X,
              const py::array_t<double, py::array::c_style>& cell_geometry)
           {
             std::array<std::size_t, 2> s_x;
             std::copy_n(X.shape(), 2, s_x.begin());
             auto _X = xt::adapt(X.data(), X.size(), xt::no_ownership(), s_x);

             std::array<std::size_t, 2> s_g;
             std::copy_n(cell_geometry.shape(), 2, s_g.begin());
             auto g = xt::adapt(cell_geometry.data(), cell_geometry.size(),
                                xt::no_ownership(), s_g);

             xt::xtensor<double, 2> x = xt::empty<double>(
                 {_X.shape(0), std::size_t(cell_geometry.shape(1))});
             const xt::xtensor<double, 2> phi
                 = xt::view(self.tabulate(0, _X), 0, xt::all(), xt::all(), 0);

             self.push_forward(x, g, phi);
             return xt_as_pyarray(std::move(x));
           })
      .def("pull_back",
           [](const dolfinx::fem::CoordinateElement& self,
              const py::array_t<double, py::array::c_style>& x,
              const py::array_t<double, py::array::c_style>& cell_geometry)
           {
             const std::size_t num_points = x.shape(0);
             const std::size_t gdim = x.shape(1);
             const std::size_t tdim
                 = dolfinx::mesh::cell_dim(self.cell_shape());

             xt::xtensor<double, 2> X = xt::empty<double>({num_points, tdim});

             std::array<std::size_t, 2> s_x;
             std::copy_n(x.shape(), 2, s_x.begin());
             auto _x = xt::adapt(x.data(), x.size(), xt::no_ownership(), s_x);

             std::array<std::size_t, 2> s_g;
             std::copy_n(cell_geometry.shape(), 2, s_g.begin());
             auto g = xt::adapt(cell_geometry.data(), cell_geometry.size(),
                                xt::no_ownership(), s_g);

             if (self.is_affine())
             {
               xt::xtensor<double, 2> J = xt::zeros<double>({gdim, tdim});
               xt::xtensor<double, 2> K = xt::zeros<double>({tdim, gdim});
               xt::xtensor<double, 4> data(self.tabulate_shape(1, 1));
               const xt::xtensor<double, 2> X0
                   = xt::zeros<double>({std::size_t(1), tdim});
               self.tabulate(1, X0, data);
               xt::xtensor<double, 2> dphi
                   = xt::view(data, xt::range(1, tdim + 1), 0, xt::all(), 0);
               self.compute_jacobian(dphi, g, J);
               self.compute_jacobian_inverse(J, K);
               self.pull_back_affine(X, K, self.x0(g), _x);
             }
             else
               self.pull_back_nonaffine(X, _x, g);

             return xt_as_pyarray(std::move(X));
           });

  py::enum_<dolfinx::fem::IntegralType>(m, "IntegralType")
      .value("cell", dolfinx::fem::IntegralType::cell)
      .value("exterior_facet", dolfinx::fem::IntegralType::exterior_facet)
      .value("interior_facet", dolfinx::fem::IntegralType::interior_facet)
      .value("vertex", dolfinx::fem::IntegralType::vertex);

  m.def(
      "locate_dofs_topological",
      [](const std::vector<
             std::reference_wrapper<const dolfinx::fem::FunctionSpace>>& V,
         int dim, const py::array_t<std::int32_t, py::array::c_style>& entities,
         bool remote) -> std::array<py::array, 2>
      {
        if (V.size() != 2)
          throw std::runtime_error("Expected two function spaces.");
        std::array<std::vector<std::int32_t>, 2> dofs
            = dolfinx::fem::locate_dofs_topological(
                {V[0], V[1]}, dim, xtl::span(entities.data(), entities.size()),
                remote);
        return {as_pyarray(std::move(dofs[0])), as_pyarray(std::move(dofs[1]))};
      },
      py::arg("V"), py::arg("dim"), py::arg("entities"),
      py::arg("remote") = true);
  m.def(
      "locate_dofs_topological",
      [](const dolfinx::fem::FunctionSpace& V, int dim,
         const py::array_t<std::int32_t, py::array::c_style>& entities,
         bool remote)
      {
        return as_pyarray(dolfinx::fem::locate_dofs_topological(
            V, dim, xtl::span(entities.data(), entities.size()), remote));
      },
      py::arg("V"), py::arg("dim"), py::arg("entities"),
      py::arg("remote") = true);
  m.def(
      "locate_dofs_geometrical",
      [](const std::vector<
             std::reference_wrapper<const dolfinx::fem::FunctionSpace>>& V,
         const std::function<py::array_t<bool>(const py::array_t<double>&)>&
             marker) -> std::array<py::array, 2>
      {
        if (V.size() != 2)
          throw std::runtime_error("Expected two function spaces.");

        auto _marker
            = [&marker](const xt::xtensor<double, 2>& x) -> xt::xtensor<bool, 1>
        {
          auto strides = x.strides();
          std::transform(strides.begin(), strides.end(), strides.begin(),
                         [](auto s) { return s * sizeof(double); });
          py::array_t _x(x.shape(), strides, x.data(), py::none());
          py::array_t m = marker(_x);
          std::vector<std::size_t> s(m.shape(), m.shape() + m.ndim());
          return xt::adapt(m.data(), m.size(), xt::no_ownership(), s);
        };

        std::array<std::vector<std::int32_t>, 2> dofs
            = dolfinx::fem::locate_dofs_geometrical({V[0], V[1]}, _marker);
        return {as_pyarray(std::move(dofs[0])), as_pyarray(std::move(dofs[1]))};
      },
      py::arg("V"), py::arg("marker"));
  m.def(
      "locate_dofs_geometrical",
      [](const dolfinx::fem::FunctionSpace& V,
         const std::function<py::array_t<bool>(const py::array_t<double>&)>&
             marker)
      {
        auto _marker
            = [&marker](const xt::xtensor<double, 2>& x) -> xt::xtensor<bool, 1>
        {
          auto strides = x.strides();
          std::transform(strides.begin(), strides.end(), strides.begin(),
                         [](auto s) { return s * sizeof(double); });
          py::array_t _x(x.shape(), strides, x.data(), py::none());
          py::array_t m = marker(_x);
          std::vector<std::size_t> s(m.shape(), m.shape() + m.ndim());
          return xt::adapt(m.data(), m.size(), xt::no_ownership(), s);
        };
        return as_pyarray(dolfinx::fem::locate_dofs_geometrical(V, _marker));
      },
      py::arg("V"), py::arg("marker"));

  // dolfinx::fem::FunctionSpace
  py::class_<dolfinx::fem::FunctionSpace,
             std::shared_ptr<dolfinx::fem::FunctionSpace>>(m, "FunctionSpace")
      .def(py::init<std::shared_ptr<dolfinx::mesh::Mesh>,
                    std::shared_ptr<dolfinx::fem::FiniteElement>,
                    std::shared_ptr<dolfinx::fem::DofMap>>())
      .def("collapse", &dolfinx::fem::FunctionSpace::collapse)
      .def("component", &dolfinx::fem::FunctionSpace::component)
      .def("contains", &dolfinx::fem::FunctionSpace::contains)
      .def_property_readonly("element", &dolfinx::fem::FunctionSpace::element)
      .def_property_readonly("mesh", &dolfinx::fem::FunctionSpace::mesh)
      .def_property_readonly("dofmap", &dolfinx::fem::FunctionSpace::dofmap)
      .def("sub", &dolfinx::fem::FunctionSpace::sub)
      .def("tabulate_dof_coordinates",
           [](const dolfinx::fem::FunctionSpace& self)
           {
             std::vector x = self.tabulate_dof_coordinates(false);
             std::vector<std::size_t> shape = {x.size() / 3, 3};
             return as_pyarray(std::move(x), shape);
           });
}
} // namespace dolfinx_wrappers
