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
#include <dolfinx/fem/dofmapbuilder.h>
#include <dolfinx/fem/interpolate.h>
#include <dolfinx/fem/sparsitybuild.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/graph/ordering.h>
#include <dolfinx/la/SparsityPattern.h>
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
#include <span>
#include <string>
#include <ufcx.h>
#include <utility>

namespace py = pybind11;

namespace
{
template <typename T, typename = void>
struct geom_type
{
  typedef T value_type;
};
template <typename T>
struct geom_type<T, std::void_t<typename T::value_type>>
{
  typedef typename T::value_type value_type;
};

// Declare DirichletBC objects for type T
template <typename T>
void declare_objects(py::module& m, const std::string& type)
{
  // dolfinx::fem::DirichletBC
  std::string pyclass_name = std::string("DirichletBC_") + type;
  py::class_<dolfinx::fem::DirichletBC<T, double>,
             std::shared_ptr<dolfinx::fem::DirichletBC<T, double>>>
      dirichletbc(m, pyclass_name.c_str(),
                  "Object for representing Dirichlet (essential) boundary "
                  "conditions");

  dirichletbc
      .def(
          py::init(
              [](const py::array_t<T, py::array::c_style>& g,
                 const py::array_t<std::int32_t, py::array::c_style>& dofs,
                 std::shared_ptr<const dolfinx::fem::FunctionSpace<double>>
V)
              {
                if (dofs.ndim() != 1)
                  throw std::runtime_error("Wrong number of dims");
                std::vector<std::size_t> shape(g.shape(), g.shape() + g.ndim());
                auto _g = std::make_shared<dolfinx::fem::Constant<T>>(
                    std::span(g.data(), g.size()), shape);
                return dolfinx::fem::DirichletBC<T, double>(
                    _g, std::vector(dofs.data(), dofs.data() + dofs.size()), V);
              }),
          py::arg("g").noconvert(), py::arg("dofs").noconvert(), py::arg("V"))
      .def(py::init(
               [](std::shared_ptr<const dolfinx::fem::Constant<T>> g,
                  const py::array_t<std::int32_t, py::array::c_style>& dofs,
                  std::shared_ptr<const dolfinx::fem::FunctionSpace<double>>
V)
               {
                 return dolfinx::fem::DirichletBC<T, double>(
                     g, std::vector(dofs.data(), dofs.data() + dofs.size()), V);
               }),
           py::arg("g").noconvert(), py::arg("dofs").noconvert(), py::arg("V"))
      .def(py::init(
               [](std::shared_ptr<const dolfinx::fem::Function<T, double>
> g,
                  const py::array_t<std::int32_t, py::array::c_style>& dofs)
               {
                 return dolfinx::fem::DirichletBC<T, double>(
                     g, std::vector(dofs.data(), dofs.data() + dofs.size()));
               }),
           py::arg("g").noconvert(), py::arg("dofs"))
      .def(
          py::init(
              [](std::shared_ptr<const dolfinx::fem::Function<T, double>
> g,
                 const std::array<py::array_t<std::int32_t, py::array::c_style>,
                                  2>& V_g_dofs,
                 std::shared_ptr<const dolfinx::fem::FunctionSpace<double>>
V)
              {
                std::array dofs
                    = {std::vector(V_g_dofs[0].data(),
                                   V_g_dofs[0].data() + V_g_dofs[0].size()),
                       std::vector(V_g_dofs[1].data(),
                                   V_g_dofs[1].data() + V_g_dofs[1].size())};
                return dolfinx::fem::DirichletBC(g, std::move(dofs), V);
              }),
          py::arg("g").noconvert(), py::arg("dofs").noconvert(),
          py::arg("V").noconvert())
      .def_property_readonly("dtype", [](const dolfinx::fem::Form<T, double>
& self)
                             { return py::dtype::of<T>(); })
      .def("dof_indices",
           [](const dolfinx::fem::DirichletBC<T, double>& self)
           {
             auto [dofs, owned] = self.dof_indices();
             return std::pair(py::array_t<std::int32_t>(
                                  dofs.size(), dofs.data(), py::cast(self)),
                              owned);
           })
      .def_property_readonly("function_space",
                             &dolfinx::fem::DirichletBC<T, double>::function_space)
      .def_property_readonly("value", &dolfinx::fem::DirichletBC<T, double>::value);

  // dolfinx::fem::Function
  std::string pyclass_name_function = std::string("Function_") + type;
  py::class_<dolfinx::fem::Function<T, double>,
             std::shared_ptr<dolfinx::fem::Function<T, double>>>(
      m, pyclass_name_function.c_str(), "A finite element function")
      .def(py::init<
               std::shared_ptr<const dolfinx::fem::FunctionSpace<double>>>(),
           "Create a function on the given function space")
      .def(py::init<std::shared_ptr<dolfinx::fem::FunctionSpace<double>>,
                    std::shared_ptr<dolfinx::la::Vector<T>>>())
      .def_readwrite("name", &dolfinx::fem::Function<T, double>::name)
      .def("sub", &dolfinx::fem::Function<T, double>::sub,
           "Return sub-function (view into parent Function")
      .def("collapse", &dolfinx::fem::Function<T, double>::collapse,
           "Collapse sub-function view")
      .def(
          "interpolate",
          [](dolfinx::fem::Function<T, double>& self,
             const py::array_t<T, py::array::c_style>& f,
             const py::array_t<std::int32_t, py::array::c_style>& cells)
          {
            if (f.ndim() == 1)
            {
              std::array<std::size_t, 2> fshape
                  = {1, static_cast<std::size_t>(f.shape(0))};
              dolfinx::fem::interpolate(self, std::span(f.data(), f.size()),
                                        fshape,
                                        std::span(cells.data(), cells.size()));
            }
            else
            {
              std::array<std::size_t, 2> fshape
                  = {static_cast<std::size_t>(f.shape(0)),
                     static_cast<std::size_t>(f.shape(1))};
              dolfinx::fem::interpolate(self, std::span(f.data(), f.size()),
                                        fshape,
                                        std::span(cells.data(), cells.size()));
            }
          },
          py::arg("f"), py::arg("cells"), "Interpolate an expression function")
      .def(
          "interpolate",
          [](dolfinx::fem::Function<T, double>& self,
             dolfinx::fem::Function<T, double>& u,
             const py::array_t<std::int32_t, py::array::c_style>& cells,
             const std::tuple<std::vector<std::int32_t>,
                              std::vector<std::int32_t>, std::vector<double>,
                              std::vector<std::int32_t>>& interpolation_data)
          {
            self.interpolate(u, std::span(cells.data(), cells.size()),
                             interpolation_data);
          },
          py::arg("u"), py::arg("cells"), py::arg("nmm_interpolation_data"),
          "Interpolate a finite element function")
      .def(
          "interpolate_ptr",
          [](dolfinx::fem::Function<T, double>& self, std::uintptr_t addr,
             const py::array_t<std::int32_t, py::array::c_style>& cells)
          {
            assert(self.function_space());
            auto element = self.function_space()->element();
            assert(element);

            // Compute value size
            auto vshape = element->value_shape();
            std::size_t value_size = std::reduce(vshape.begin(), vshape.end(),
                                                 1, std::multiplies{});

            assert(self.function_space()->mesh());
            const std::vector<double> x = dolfinx::fem::interpolation_coords(
                *element, self.function_space()->mesh()->geometry(),
                std::span(cells.data(), cells.size()));

            std::array<std::size_t, 2> shape = {value_size, x.size() / 3};
            std::vector<T> values(shape[0] * shape[1]);
            std::function<void(T*, int, int, const double*)> f
                = reinterpret_cast<void (*)(T*, int, int, const double*)>(addr);
            f(values.data(), shape[1], shape[0], x.data());

            dolfinx::fem::interpolate(self, std::span<const T>(values), shape,
                                      std::span(cells.data(), cells.size()));
          },
          py::arg("f_ptr"), py::arg("cells"),
          "Interpolate using a pointer to an expression with a C signature")
      .def(
          "interpolate",
          [](dolfinx::fem::Function<T, double>& self,
             const dolfinx::fem::Expression<T, double>& expr,
             const py::array_t<std::int32_t, py::array::c_style>& cells)
          { self.interpolate(expr, std::span(cells.data(), cells.size())); },
          py::arg("expr"), py::arg("cells"),
          "Interpolate an Expression on a set of cells")
      .def_property_readonly(
          "x", py::overload_cast<>(&dolfinx::fem::Function<T, double>::x),
          "Return the vector associated with the finite element Function")
      .def(
          "eval",
          [](const dolfinx::fem::Function<T, double>& self,
             const py::array_t<double, py::array::c_style>& x,
             const py::array_t<std::int32_t, py::array::c_style>& cells,
             py::array_t<T, py::array::c_style>& u)
          {
            // TODO: handle 1d case
            self.eval(std::span(x.data(), x.size()),
                      {static_cast<std::size_t>(x.shape(0)),
                       static_cast<std::size_t>(x.shape(1))},
                      std::span(cells.data(), cells.size()),
                      std::span(u.mutable_data(), u.size()),
                      {static_cast<std::size_t>(u.shape(0)),
                       static_cast<std::size_t>(u.shape(1))});
          },
          py::arg("x"), py::arg("cells"), py::arg("values"),
          "Evaluate Function")
      .def_property_readonly(
          "function_space", &dolfinx::fem::Function<T, double>::function_space);

  // dolfinx::fem::Constant
  std::string pyclass_name_constant = std::string("Constant_") + type;
  py::class_<dolfinx::fem::Constant<T>,
             std::shared_ptr<dolfinx::fem::Constant<T>>>(
      m, pyclass_name_constant.c_str(),
      "Value constant with respect to integration domain")
      .def(py::init(
               [](const py::array_t<T, py::array::c_style>& c)
               {
                 std::vector<std::size_t> shape(c.shape(),
                                                c.shape() + c.ndim());
                 return dolfinx::fem::Constant<T>(std::span(c.data(), c.size()),
                                                  shape);
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
      class_<dolfinx::fem::Expression<T, double>,
             std::shared_ptr<dolfinx::fem::Expression<T, double>>>(
          m, pyclass_name_expr.c_str(), "An Expression")
          .def(py::init(
                   [](const std::vector<std::shared_ptr<
                          const dolfinx::fem::Function<T, double>
>>& coefficients,
                      const std::vector<std::shared_ptr<
                          const dolfinx::fem::Constant<T>>>& constants,
                      const py::array_t<double, py::array::c_style>& X,
                      std::uintptr_t fn_addr,
                      const std::vector<int>& value_shape,
                      std::shared_ptr<const dolfinx::mesh::Mesh<double>> mesh,
                      std::shared_ptr<const dolfinx::fem::FunctionSpace<double>>
                          argument_function_space)
                   {
                     auto tabulate_expression_ptr
                         = (void (*)(T*, const T*, const T*,
                                     const typename geom_type<T>::value_type*,
                                     const int*, const std::uint8_t*))fn_addr;
                     return dolfinx::fem::Expression<T, double>(
                         coefficients, constants, std::span(X.data(), X.size()),
                         {static_cast<std::size_t>(X.shape(0)),
                          static_cast<std::size_t>(X.shape(1))},
                         tabulate_expression_ptr, value_shape, mesh,
                         argument_function_space);
                   }),
               py::arg("coefficients"), py::arg("constants"), py::arg("X"),
               py::arg("fn"), py::arg("value_shape"), py::arg("mesh"),
               py::arg("argument_function_space"))
          .def(
              "eval",
              [](const dolfinx::fem::Expression<T, double>& self,
                 const py::array_t<std::int32_t,
                                   py::array::c_style>& active_cells,
                 py::array_t<T, py::array::c_style>& values)
              {
                self.eval(std::span(active_cells.data(), active_cells.size()),
                          std::span(values.mutable_data(), values.size()),
                          {(std::size_t)values.shape(0), (std::size_t)values.shape(1)});
              },
              py::arg("active_cells"), py::arg("values"))
          .def("X",
               [](const dolfinx::fem::Expression<T, double>& self)
               {
                 auto [X, shape] = self.X();
                 return dolfinx_wrappers::as_pyarray(std::move(X), shape);
               })
          .def_property_readonly("dtype",
                                 [](const dolfinx::fem::Expression<T, double>& self)
                                 { return py::dtype::of<T>(); })
          .def_property_readonly("mesh", &dolfinx::fem::Expression<T, double>::mesh)
          .def_property_readonly("value_size",
                                 &dolfinx::fem::Expression<T, double>::value_size)
          .def_property_readonly("value_shape",
                                 &dolfinx::fem::Expression<T, double>::value_shape);

  std::string pymethod_create_expression
      = std::string("create_expression_") + type;
  m.def(
      pymethod_create_expression.c_str(),
      [](const std::uintptr_t expression,
         const std::vector<std::shared_ptr<
             const dolfinx::fem::Function<T, double>>>& coefficients,
         const std::vector<std::shared_ptr<const dolfinx::fem::Constant<T>>>&
             constants,
         std::shared_ptr<const dolfinx::mesh::Mesh<double>> mesh,
         std::shared_ptr<const dolfinx::fem::FunctionSpace<double>>
             argument_function_space)
      {
        const ufcx_expression* p
            = reinterpret_cast<const ufcx_expression*>(expression);
        return dolfinx::fem::create_expression<T, double>(
            *p, coefficients, constants, mesh, argument_function_space);
      },
      py::arg("expression"), py::arg("coefficients"), py::arg("constants"),
      py::arg("mesh"), py::arg("argument_function_space"),
      "Create Form from a pointer to ufc_form.");
}

template <typename T>
void declare_form(py::module& m, const std::string& type)
{
  // dolfinx::fem::Form
  std::string pyclass_name_form = std::string("Form_") + type;
  py::class_<dolfinx::fem::Form<T, double>,
             std::shared_ptr<dolfinx::fem::Form<T, double>>>(
      m, pyclass_name_form.c_str(), "Variational form object")
      .def(py::init(
               [](const std::vector<std::shared_ptr<
                      const dolfinx::fem::FunctionSpace<double>>>& spaces,
                  const std::map<
                      dolfinx::fem::IntegralType,
                      std::vector<std::tuple<int, py::object,
                                             py::array_t<std::int32_t>>>>&
                      integrals,
                  const std::vector<std::shared_ptr<
                      const dolfinx::fem::Function<T, double>>>& coefficients,
                  const std::vector<std::shared_ptr<
                      const dolfinx::fem::Constant<T>>>& constants,
                  bool needs_permutation_data,
                  std::shared_ptr<const dolfinx::mesh::Mesh<double>> mesh)
               {
                 using kern = std::function<void(
                     T*, const T*, const T*,
                     const typename geom_type<T>::value_type*, const int*,
                     const std::uint8_t*)>;
                 std::map<dolfinx::fem::IntegralType,
                          std::vector<
                              std::tuple<int, kern, std::vector<std::int32_t>>>>
                     _integrals;

                 // Loop over kernel for each entity type
                 for (auto& [type, kernels] : integrals)
                 {
                   for (auto& [id, kn, e] : kernels)
                   {
                     std::uintptr_t ptr = kn.template cast<std::uintptr_t>();
                     auto kn_ptr
                         = (void (*)(T*, const T*, const T*,
                                     const typename geom_type<T>::value_type*,
                                     const int*, const std::uint8_t*))ptr;
                     _integrals[type].emplace_back(
                         id, kn_ptr,
                         std::vector<std::int32_t>(e.data(),
                                                   e.data() + e.size()));
                   }
                 }

                 return dolfinx::fem::Form<T, double>(
                     spaces, _integrals, coefficients, constants,
                     needs_permutation_data, mesh);
               }),
           py::arg("spaces"), py::arg("integrals"), py::arg("coefficients"),
           py::arg("constants"), py::arg("need_permutation_data"),
           py::arg("mesh") = py::none())
      .def(py::init(
               [](std::uintptr_t form,
                  const std::vector<std::shared_ptr<
                      const dolfinx::fem::FunctionSpace<double>>>& spaces,
                  const std::vector<std::shared_ptr<
                      const dolfinx::fem::Function<T, double>>>& coefficients,
                  const std::vector<std::shared_ptr<
                      const dolfinx::fem::Constant<T>>>& constants,
                  const std::map<dolfinx::fem::IntegralType,
                                 std::vector<std::pair<
                                     std::int32_t,
                                     py::array_t<std::int32_t,
                                                 py::array::c_style
                                                     | py::array::forcecast>>>>&
                      subdomains,
                  std::shared_ptr<const dolfinx::mesh::Mesh<double>> mesh)
               {
                 std::map<dolfinx::fem::IntegralType,
                          std::vector<std::pair<std::int32_t,
                                                std::vector<std::int32_t>>>>
                     sd;
                 for (auto& [itg, data] : subdomains)
                 {
                   std::vector<
                       std::pair<std::int32_t, std::vector<std::int32_t>>>
                       x;
                   for (auto& [id, e] : data)
                   {
                     x.emplace_back(id,
                                    std::vector(e.data(), e.data() + e.size()));
                   }
                   sd.insert({itg, std::move(x)});
                 }

                 ufcx_form* p = reinterpret_cast<ufcx_form*>(form);
                 return dolfinx::fem::create_form<T>(*p, spaces, coefficients,
                                                     constants, sd, mesh);
               }),
           py::arg("form"), py::arg("spaces"), py::arg("coefficients"),
           py::arg("constants"), py::arg("subdomains"), py::arg("mesh"),
           "Create a Form from a pointer to a ufcx_form")
      .def_property_readonly("dtype",
                             [](const dolfinx::fem::Form<T, double>& self)
                             { return py::dtype::of<T>(); })
      .def_property_readonly("coefficients",
                             &dolfinx::fem::Form<T, double>::coefficients)
      .def_property_readonly("rank", &dolfinx::fem::Form<T, double>::rank)
      .def_property_readonly("mesh", &dolfinx::fem::Form<T, double>::mesh)
      .def_property_readonly("function_spaces",
                             &dolfinx::fem::Form<T, double>::function_spaces)
      .def("integral_ids", &dolfinx::fem::Form<T, double>::integral_ids)
      .def_property_readonly("integral_types",
                             &dolfinx::fem::Form<T, double>::integral_types)
      .def_property_readonly(
          "needs_facet_permutations",
          &dolfinx::fem::Form<T, double>::needs_facet_permutations)
      .def(
          "domains",
          [](const dolfinx::fem::Form<T, double>& self,
             dolfinx::fem::IntegralType type,
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
              const std::vector<std::int32_t>& _d
                  = self.exterior_facet_domains(i);
              std::array<py::ssize_t, 2> shape
                  = {py::ssize_t(_d.size()) / 2, 2};
              return py::array_t<std::int32_t>(shape, _d.data(),
                                               py::cast(self));
            }
            case dolfinx::fem::IntegralType::interior_facet:
            {
              const std::vector<std::int32_t>& _d
                  = self.interior_facet_domains(i);
              std::array<py::ssize_t, 3> shape
                  = {py::ssize_t(_d.size()) / 4, 2, 2};
              return py::array_t<std::int32_t>(shape, _d.data(),
                                               py::cast(self));
            }
            default:
              throw ::std::runtime_error("Integral type unsupported.");
            }
          },
          py::arg("type"), py::arg("i"));

  // Form
  std::string pymethod_create_form = std::string("create_form_") + type;
  m.def(
      pymethod_create_form.c_str(),
      [](std::uintptr_t form,
         const std::vector<std::shared_ptr<
             const dolfinx::fem::FunctionSpace<double>>>& spaces,
         const std::vector<std::shared_ptr<
             const dolfinx::fem::Function<T, double>>>& coefficients,
         const std::vector<std::shared_ptr<const dolfinx::fem::Constant<T>>>&
             constants,
         const std::map<
             dolfinx::fem::IntegralType,
             std::vector<std::pair<
                 std::int32_t,
                 py::array_t<std::int32_t,
                             py::array::c_style | py::array::forcecast>>>>&
             subdomains,
         std::shared_ptr<const dolfinx::mesh::Mesh<double>> mesh)
      {
        std::map<
            dolfinx::fem::IntegralType,
            std::vector<std::pair<std::int32_t, std::vector<std::int32_t>>>>
            sd;
        for (auto& [itg, data] : subdomains)
        {
          std::vector<std::pair<std::int32_t, std::vector<std::int32_t>>> x;
          for (auto& [id, idx] : data)
          {
            x.emplace_back(id,
                           std::vector(idx.data(), idx.data() + idx.size()));
          }
          sd.insert({itg, std::move(x)});
        }

        ufcx_form* p = reinterpret_cast<ufcx_form*>(form);
        return dolfinx::fem::create_form<T>(*p, spaces, coefficients, constants,
                                            sd, mesh);
      },
      py::arg("form"), py::arg("spaces"), py::arg("coefficients"),
      py::arg("constants"), py::arg("subdomains"), py::arg("mesh"),
      "Create Form from a pointer to ufcx_form.");
}
} // namespace

namespace dolfinx_wrappers
{

void fem(py::module& m)
{
  // Load basix and dolfinx to use Pybindings
  py::module_::import("basix");

  declare_objects<float>(m, "float32");
  declare_objects<double>(m, "float64");
  declare_objects<std::complex<float>>(m, "complex64");
  declare_objects<std::complex<double>>(m, "complex128");

  declare_form<float>(m, "float32");
  declare_form<double>(m, "float64");
  declare_form<std::complex<float>>(m, "complex64");
  declare_form<std::complex<double>>(m, "complex128");

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
      py::arg("topology"), py::arg("dofmaps"), py::arg("types"),
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
      py::arg("dofmap"), py::arg("cell_type"), py::arg("parent_map"),
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
      py::arg("comm"), py::arg("dofmap"), py::arg("topology"),
      py::arg("element"),
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
      py::arg("comm"), py::arg("topology"), py::arg("layout"),
      "Build and dofmap on a mesh.");
  m.def("transpose_dofmap", &dolfinx::fem::transpose_dofmap,
        "Build the index to (cell, local index) map from a "
        "dofmap ((cell, local index ) -> index).");
  m.def(
      "compute_integration_domains",
      [](dolfinx::fem::IntegralType type,
         const dolfinx::mesh::MeshTags<int>& meshtags)
      {
        return dolfinx::fem::compute_integration_domains(
            type, meshtags.topology(), meshtags.indices(), meshtags.dim(),
            meshtags.values());
      },
      py::arg("integral_type"), py::arg("meshtags"));
  m.def(
      "create_nonmatching_meshes_interpolation_data",
      [](const dolfinx::mesh::Mesh<double>& mesh0,
         const dolfinx::fem::FiniteElement& element0,
         const dolfinx::mesh::Mesh<double>& mesh1)
      {
        int tdim = mesh0.topology().dim();
        auto cell_map = mesh0.topology().index_map(tdim);
        assert(cell_map);
        std::int32_t num_cells
            = cell_map->size_local() + cell_map->num_ghosts();
        std::vector<std::int32_t> cells(num_cells, 0);
        std::iota(cells.begin(), cells.end(), 0);
        return dolfinx::fem::create_nonmatching_meshes_interpolation_data(
            mesh0.geometry(), element0, mesh1,
            std::span(cells.data(), cells.size()));
      },
      py::arg("mesh0"), py::arg("element0"), py::arg("mesh1"));
  m.def(
      "create_nonmatching_meshes_interpolation_data",
      [](const dolfinx::mesh::Geometry<double>& geometry0,
         const dolfinx::fem::FiniteElement& element0,
         const dolfinx::mesh::Mesh<double>& mesh1,
         const py::array_t<std::int32_t, py::array::c_style>& cells)
      {
        return dolfinx::fem::create_nonmatching_meshes_interpolation_data(
            geometry0, element0, mesh1, std::span(cells.data(), cells.size()));
      },
      py::arg("geometry0"), py::arg("element0"), py::arg("mesh1"),
      py::arg("cells"));

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
               }),
           py::arg("ufcx_element"))
      .def("__eq__", &dolfinx::fem::FiniteElement::operator==)
      .def_property_readonly("basix_element",
                             &dolfinx::fem::FiniteElement::basix_element,
                             py::return_value_policy::reference_internal)
      .def_property_readonly("num_sub_elements",
                             &dolfinx::fem::FiniteElement::num_sub_elements)
      .def("interpolation_points",
           [](const dolfinx::fem::FiniteElement& self)
           {
             auto [X, shape] = self.interpolation_points();
             return as_pyarray(std::move(X), shape);
           })
      .def_property_readonly("interpolation_ident",
                             &dolfinx::fem::FiniteElement::interpolation_ident)
      .def_property_readonly("space_dimension",
                             &dolfinx::fem::FiniteElement::space_dimension)
      .def_property_readonly(
          "value_shape",
          [](const dolfinx::fem::FiniteElement& self)
          {
            std::span<const std::size_t> shape = self.value_shape();
            return py::array_t(shape.size(), shape.data(), py::none());
          })
      .def(
          "apply_dof_transformation",
          [](const dolfinx::fem::FiniteElement& self,
             py::array_t<double, py::array::c_style> x,
             std::uint32_t cell_permutation, int dim)
          {
            self.apply_dof_transformation(std::span(x.mutable_data(), x.size()),
                                          cell_permutation, dim);
          },
          py::arg("x"), py::arg("cell_permutation"), py::arg("dim"))
      .def(
          "apply_transpose_dof_transformation",
          [](const dolfinx::fem::FiniteElement& self,
             py::array_t<double, py::array::c_style> x,
             std::uint32_t cell_permutation, int dim)
          {
            self.apply_transpose_dof_transformation(
                std::span(x.mutable_data(), x.size()), cell_permutation, dim);
          },
          py::arg("x"), py::arg("cell_permutation"), py::arg("dim"))
      .def(
          "apply_inverse_transpose_dof_transformation",
          [](const dolfinx::fem::FiniteElement& self,
             py::array_t<double, py::array::c_style> x,
             std::uint32_t cell_permutation, int dim)
          {
            self.apply_inverse_transpose_dof_transformation(
                std::span(x.mutable_data(), x.size()), cell_permutation, dim);
          },
          py::arg("x"), py::arg("cell_permutation"), py::arg("dim"))
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
                    const std::vector<dolfinx::fem::ElementDofLayout>&>(),
           py::arg("block_size"), py::arg("endity_dofs"),
           py::arg("entity_closure_dofs"), py::arg("parent_map"),
           py::arg("sub_layouts"))
      .def_property_readonly("num_dofs",
                             &dolfinx::fem::ElementDofLayout::num_dofs)
      .def("num_entity_dofs", &dolfinx::fem::ElementDofLayout::num_entity_dofs,
           py::arg("dim"))
      .def("num_entity_closure_dofs",
           &dolfinx::fem::ElementDofLayout::num_entity_closure_dofs,
           py::arg("dim"))
      .def("entity_dofs", &dolfinx::fem::ElementDofLayout::entity_dofs,
           py::arg("dim"), py::arg("entity_index"))
      .def("entity_closure_dofs",
           &dolfinx::fem::ElementDofLayout::entity_closure_dofs, py::arg("dim"),
           py::arg("entity_index"))
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
      .def(
          "cell_dofs",
          [](const dolfinx::fem::DofMap& self, int cell)
          {
            std::span<const std::int32_t> dofs = self.cell_dofs(cell);
            return py::array_t<std::int32_t>(dofs.size(), dofs.data(),
                                             py::cast(self));
          },
          py::arg("cell"))
      .def_property_readonly("bs", &dolfinx::fem::DofMap::bs)
      .def("list", &dolfinx::fem::DofMap::list,
           py::return_value_policy::reference_internal);

  // dolfinx::fem::CoordinateElement
  py::class_<dolfinx::fem::CoordinateElement,
             std::shared_ptr<dolfinx::fem::CoordinateElement>>(
      m, "CoordinateElement", "Coordinate map element")
      .def(py::init<dolfinx::mesh::CellType, int>(), py::arg("celltype"),
           py::arg("degree"))
      .def(py::init<dolfinx::mesh::CellType, int,
                    basix::element::lagrange_variant>(),
           py::arg("celltype"), py::arg("degree"), py::arg("variant"))
      .def("create_dof_layout",
           &dolfinx::fem::CoordinateElement::create_dof_layout)
      .def_property_readonly("degree", &dolfinx::fem::CoordinateElement::degree)
      .def_property_readonly("variant",
                             &dolfinx::fem::CoordinateElement::variant)
      .def(
          "push_forward",
          [](const dolfinx::fem::CoordinateElement& self,
             const py::array_t<double, py::array::c_style>& X,
             const py::array_t<double, py::array::c_style>& cell)
          {
            namespace stdex = std::experimental;
            using mdspan2_t
                = stdex::mdspan<double, stdex::dextents<std::size_t, 2>>;
            using cmdspan2_t
                = stdex::mdspan<const double, stdex::dextents<std::size_t, 2>>;
            using cmdspan4_t
                = stdex::mdspan<const double, stdex::dextents<std::size_t, 4>>;

            std::array<std::size_t, 2> Xshape
                = {(std::size_t)X.shape(0), (std::size_t)X.shape(1)};

            std::array<std::size_t, 4> phi_shape
                = self.tabulate_shape(0, X.shape(0));
            std::vector<double> phi_b(std::reduce(
                phi_shape.begin(), phi_shape.end(), 1, std::multiplies{}));
            cmdspan4_t phi_full(phi_b.data(), phi_shape);
            self.tabulate(0, std::span(X.data(), X.size()), Xshape, phi_b);
            auto phi = stdex::submdspan(phi_full, 0, stdex::full_extent,
                                        stdex::full_extent, 0);

            std::array<std::size_t, 2> shape
                = {(std::size_t)X.shape(0), (std::size_t)cell.shape(1)};
            std::vector<double> xb(shape[0] * shape[1]);
            self.push_forward(
                mdspan2_t(xb.data(), shape),
                cmdspan2_t(cell.data(), cell.shape(0), cell.shape(1)), phi);

            return as_pyarray(std::move(xb), shape);
          },
          py::arg("X"), py::arg("cell_geometry"))
      .def(
          "pull_back",
          [](const dolfinx::fem::CoordinateElement& self,
             const py::array_t<double, py::array::c_style>& x,
             const py::array_t<double, py::array::c_style>& cell_geometry)
          {
            const std::size_t num_points = x.shape(0);
            const std::size_t gdim = x.shape(1);
            const std::size_t tdim = dolfinx::mesh::cell_dim(self.cell_shape());

            namespace stdex = std::experimental;
            using mdspan2_t
                = stdex::mdspan<double, stdex::dextents<std::size_t, 2>>;
            using cmdspan2_t
                = stdex::mdspan<const double, stdex::dextents<std::size_t, 2>>;
            using cmdspan4_t
                = stdex::mdspan<const double, stdex::dextents<std::size_t, 4>>;

            std::vector<double> Xb(num_points * tdim);
            mdspan2_t X(Xb.data(), num_points, tdim);
            cmdspan2_t _x(x.data(), x.shape(0), x.shape(1));
            cmdspan2_t g(cell_geometry.data(), cell_geometry.shape(0),
                         cell_geometry.shape(1));

            if (self.is_affine())
            {
              std::vector<double> J_b(gdim * tdim);
              mdspan2_t J(J_b.data(), gdim, tdim);
              std::vector<double> K_b(tdim * gdim);
              mdspan2_t K(K_b.data(), tdim, gdim);

              std::array<std::size_t, 4> phi_shape = self.tabulate_shape(1, 1);
              std::vector<double> phi_b(std::reduce(
                  phi_shape.begin(), phi_shape.end(), 1, std::multiplies{}));
              cmdspan4_t phi(phi_b.data(), phi_shape);

              self.tabulate(1, std::vector<double>(tdim), {1, tdim}, phi_b);
              auto dphi = stdex::submdspan(phi, std::pair(1, tdim + 1), 0,
                                           stdex::full_extent, 0);

              self.compute_jacobian(dphi, g, J);
              self.compute_jacobian_inverse(J, K);
              std::array<double, 3> x0 = {0, 0, 0};
              for (std::size_t i = 0; i < g.extent(1); ++i)
                x0[i] += g(0, i);
              self.pull_back_affine(X, K, x0, _x);
            }
            else
              self.pull_back_nonaffine(X, _x, g);

            return as_pyarray(std::move(Xb), std::array{num_points, tdim});
          },
          py::arg("x"), py::arg("cell_geometry"));

  py::enum_<dolfinx::fem::IntegralType>(m, "IntegralType")
      .value("cell", dolfinx::fem::IntegralType::cell)
      .value("exterior_facet", dolfinx::fem::IntegralType::exterior_facet)
      .value("interior_facet", dolfinx::fem::IntegralType::interior_facet)
      .value("vertex", dolfinx::fem::IntegralType::vertex);

  m.def(
      "locate_dofs_topological",
      [](const std::vector<std::reference_wrapper<
             const dolfinx::fem::FunctionSpace<double>>>& V,
         int dim, const py::array_t<std::int32_t, py::array::c_style>& entities,
         bool remote) -> std::array<py::array, 2>
      {
        if (V.size() != 2)
          throw std::runtime_error("Expected two function spaces.");
        std::array<std::vector<std::int32_t>, 2> dofs
            = dolfinx::fem::locate_dofs_topological(
                V[0].get().mesh()->topology_mutable(),
                {*V[0].get().dofmap(), *V[1].get().dofmap()}, dim,
                std::span(entities.data(), entities.size()), remote);
        return {as_pyarray(std::move(dofs[0])), as_pyarray(std::move(dofs[1]))};
      },
      py::arg("V"), py::arg("dim"), py::arg("entities"),
      py::arg("remote") = true);
  m.def(
      "locate_dofs_topological",
      [](const dolfinx::fem::FunctionSpace<double>& V, int dim,
         const py::array_t<std::int32_t, py::array::c_style>& entities,
         bool remote)
      {
        return as_pyarray(dolfinx::fem::locate_dofs_topological(
            V.mesh()->topology_mutable(), *V.dofmap(), dim,
            std::span(entities.data(), entities.size()), remote));
      },
      py::arg("V"), py::arg("dim"), py::arg("entities"),
      py::arg("remote") = true);
  m.def(
      "locate_dofs_geometrical",
      [](const std::vector<std::reference_wrapper<
             const dolfinx::fem::FunctionSpace<double>>>& V,
         const std::function<py::array_t<bool>(const py::array_t<double>&)>&
             marker) -> std::array<py::array, 2>
      {
        if (V.size() != 2)
          throw std::runtime_error("Expected two function spaces.");

        auto _marker = [&marker](auto x)
        {
          std::array<std::size_t, 2> shape = {x.extent(0), x.extent(1)};
          py::array_t<double> x_view(shape, x.data_handle(), py::none());
          py::array_t<bool> marked = marker(x_view);
          return std::vector<std::int8_t>(marked.data(),
                                          marked.data() + marked.size());
        };

        std::array<std::vector<std::int32_t>, 2> dofs
            = dolfinx::fem::locate_dofs_geometrical<double>({V[0], V[1]},
                                                            _marker);
        return {as_pyarray(std::move(dofs[0])), as_pyarray(std::move(dofs[1]))};
      },
      py::arg("V"), py::arg("marker"));
  m.def(
      "locate_dofs_geometrical",
      [](const dolfinx::fem::FunctionSpace<double>& V,
         const std::function<py::array_t<bool>(const py::array_t<double>&)>&
             marker)
      {
        auto _marker = [&marker](auto x)
        {
          std::array<std::size_t, 2> shape = {x.extent(0), x.extent(1)};
          py::array_t<double> x_view(shape, x.data_handle(), py::none());
          py::array_t<bool> marked = marker(x_view);
          return std::vector<std::int8_t>(marked.data(),
                                          marked.data() + marked.size());
        };

        return as_pyarray(dolfinx::fem::locate_dofs_geometrical(V, _marker));
      },
      py::arg("V"), py::arg("marker"));

  m.def(
      "interpolation_coords",
      [](const dolfinx::fem::FiniteElement& e,
         const dolfinx::mesh::Geometry<double>& geometry,
         py::array_t<std::int32_t, py::array::c_style> cells)
      {
        std::vector<double> x = dolfinx::fem::interpolation_coords(
            e, geometry, std::span(cells.data(), cells.size()));
        return as_pyarray(std::move(x),
                          std::array<std::size_t, 2>{3, x.size() / 3});
      },
      py::arg("element"), py::arg("V"), py::arg("cells"));

  // dolfinx::fem::FunctionSpace
  py::class_<dolfinx::fem::FunctionSpace<double>,
             std::shared_ptr<dolfinx::fem::FunctionSpace<double>>>(
      m, "FunctionSpace")
      .def(py::init<std::shared_ptr<dolfinx::mesh::Mesh<double>>,
                    std::shared_ptr<dolfinx::fem::FiniteElement>,
                    std::shared_ptr<dolfinx::fem::DofMap>>(),
           py::arg("mesh"), py::arg("element"), py::arg("dofmap"))
      .def("collapse", &dolfinx::fem::FunctionSpace<double>::collapse)
      .def("component", &dolfinx::fem::FunctionSpace<double>::component)
      .def("contains", &dolfinx::fem::FunctionSpace<double>::contains,
           py::arg("V"))
      .def_property_readonly("element",
                             &dolfinx::fem::FunctionSpace<double>::element)
      .def_property_readonly("mesh", &dolfinx::fem::FunctionSpace<double>::mesh)
      .def_property_readonly("dofmap",
                             &dolfinx::fem::FunctionSpace<double>::dofmap)
      .def("sub", &dolfinx::fem::FunctionSpace<double>::sub,
           py::arg("component"))
      .def("tabulate_dof_coordinates",
           [](const dolfinx::fem::FunctionSpace<double>& self)
           {
             std::vector x = self.tabulate_dof_coordinates(false);
             std::vector<std::size_t> shape = {x.size() / 3, 3};
             return as_pyarray(std::move(x), shape);
           });
}
} // namespace dolfinx_wrappers
