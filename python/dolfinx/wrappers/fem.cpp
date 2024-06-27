// Copyright (C) 2017-2021 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "array.h"
#include "caster_mpi.h"
#include "numpy_dtype.h"
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
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/set.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>
#include <span>
#include <string>
#include <ufcx.h>
#include <utility>

namespace nb = nanobind;

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

template <typename T>
void declare_function_space(nb::module_& m, std::string type)
{
  {
    std::string pyclass_name = "FunctionSpace_" + type;
    nb::class_<dolfinx::fem::FunctionSpace<T>>(m, pyclass_name.c_str(),
                                               "Finite element function space")
        .def(nb::init<std::shared_ptr<const dolfinx::mesh::Mesh<T>>,
                      std::shared_ptr<const dolfinx::fem::FiniteElement<T>>,
                      std::shared_ptr<const dolfinx::fem::DofMap>,
                      std::vector<std::size_t>>(),
             nb::arg("mesh"), nb::arg("element"), nb::arg("dofmap"),
             nb::arg("value_shape"))
        .def("collapse", &dolfinx::fem::FunctionSpace<T>::collapse)
        .def("component", &dolfinx::fem::FunctionSpace<T>::component)
        .def("contains", &dolfinx::fem::FunctionSpace<T>::contains,
             nb::arg("V"))
        .def_prop_ro("element", &dolfinx::fem::FunctionSpace<T>::element)
        .def_prop_ro("mesh", &dolfinx::fem::FunctionSpace<T>::mesh)
        .def_prop_ro("dofmap", &dolfinx::fem::FunctionSpace<T>::dofmap)
        .def_prop_ro(
            "value_shape",
            [](const dolfinx::fem::FunctionSpace<T>& self)
            {
              std::span<const std::size_t> vshape = self.value_shape();
              return nb::ndarray<const std::size_t, nb::numpy>(
                  vshape.data(), {vshape.size()}, nb::handle());
            },
            nb::rv_policy::reference_internal)
        .def("sub", &dolfinx::fem::FunctionSpace<T>::sub, nb::arg("component"))
        .def("tabulate_dof_coordinates",
             [](const dolfinx::fem::FunctionSpace<T>& self)
             {
               std::vector x = self.tabulate_dof_coordinates(false);
               return dolfinx_wrappers::as_nbarray(std::move(x),
                                                   {x.size() / 3, 3});
             });
  }

  {
    std::string pyclass_name = "FiniteElement_" + type;
    nb::class_<dolfinx::fem::FiniteElement<T>>(m, pyclass_name.c_str(),
                                               "Finite element object")
        .def(
            "__init__",
            [](dolfinx::fem::FiniteElement<T>* self,
               basix::FiniteElement<T>& element, std::size_t block_size,
               bool symmetric) {
              new (self) dolfinx::fem::FiniteElement<T>(element, block_size,
                                                        symmetric);
            },
            nb::arg("element"), nb::arg("block_size"), nb::arg("symmetric"))
        .def(
            "__init__",
            [](dolfinx::fem::FiniteElement<T>* self,
               std::vector<
                   std::shared_ptr<const dolfinx::fem::FiniteElement<T>>>
                   elements)
            { new (self) dolfinx::fem::FiniteElement<T>(elements); },
            nb::arg("elements"))
        .def(
            "__init__",
            [](dolfinx::fem::FiniteElement<T>* self, mesh::CellType cell_type,
               nb::ndarray<T, nb::ndim<2>, nb::numpy> points,
               std::size_t block_size, bool symmetry)
            {
              std::span<T> pdata(points.data(), points.size());
              new (self) dolfinx::fem::FiniteElement<T>(
                  cell_type, pdata, {points.shape(0), points.shape(1)},
                  block_size, symmetry);
            },
            nb::arg("cell_type"), nb::arg("points"), nb::arg("block_size"),
            nb::arg("symmetry"))
        .def("__eq__", &dolfinx::fem::FiniteElement<T>::operator==)
        .def_prop_ro("dtype", [](const dolfinx::fem::FiniteElement<T>&)
                     { return dolfinx_wrappers::numpy_dtype<T>(); })
        .def_prop_ro("basix_element",
                     &dolfinx::fem::FiniteElement<T>::basix_element,
                     nb::rv_policy::reference_internal)
        .def_prop_ro("num_sub_elements",
                     &dolfinx::fem::FiniteElement<T>::num_sub_elements)
        .def("interpolation_points",
             [](const dolfinx::fem::FiniteElement<T>& self)
             {
               auto [X, shape] = self.interpolation_points();
               return dolfinx_wrappers::as_nbarray(std::move(X), shape.size(),
                                                   shape.data());
             })
        .def_prop_ro("interpolation_ident",
                     &dolfinx::fem::FiniteElement<T>::interpolation_ident)
        .def_prop_ro("space_dimension",
                     &dolfinx::fem::FiniteElement<T>::space_dimension)
        .def(
            "T_apply",
            [](const dolfinx::fem::FiniteElement<T>& self,
               nb::ndarray<T, nb::ndim<1>, nb::c_contig> x,
               nb::ndarray<const std::uint32_t, nb::ndim<1>, nb::c_contig>
                   cell_permutations,
               int dim)
            {
              const std::size_t data_per_cell
                  = x.size() / cell_permutations.size();
              std::span<T> x_span(x.data(), x.size());
              std::span<const std::uint32_t> perm_span(
                  cell_permutations.data(), cell_permutations.size());
              for (std::size_t i = 0; i < cell_permutations.size(); i++)
              {
                self.T_apply(x_span.subspan(i * data_per_cell, data_per_cell),
                             perm_span[i], dim);
              }
            },
            nb::arg("x"), nb::arg("cell_permutations"), nb::arg("dim"))
        .def(
            "Tt_apply",
            [](const dolfinx::fem::FiniteElement<T>& self,
               nb::ndarray<T, nb::ndim<1>, nb::c_contig> x,
               nb::ndarray<const std::uint32_t, nb::ndim<1>, nb::c_contig>
                   cell_permutations,
               int dim)
            {
              const std::size_t data_per_cell
                  = x.size() / cell_permutations.size();
              std::span<T> x_span(x.data(), x.size());
              std::span<const std::uint32_t> perm_span(
                  cell_permutations.data(), cell_permutations.size());
              for (std::size_t i = 0; i < cell_permutations.size(); i++)
              {
                self.Tt_apply(x_span.subspan(i * data_per_cell, data_per_cell),
                              perm_span[i], dim);
              }
            },
            nb::arg("x"), nb::arg("cell_permutations"), nb::arg("dim"))
        .def(
            "Tt_inv_apply",
            [](const dolfinx::fem::FiniteElement<T>& self,
               nb::ndarray<T, nb::ndim<1>, nb::c_contig> x,
               nb::ndarray<const std::uint32_t, nb::ndim<1>, nb::c_contig>
                   cell_permutations,
               int dim)
            {
              const std::size_t data_per_cell
                  = x.size() / cell_permutations.size();
              std::span<T> x_span(x.data(), x.size());
              std::span<const std::uint32_t> perm_span(
                  cell_permutations.data(), cell_permutations.size());

              for (std::size_t i = 0; i < cell_permutations.size(); i++)
              {
                self.Tt_inv_apply(
                    x_span.subspan(i * data_per_cell, data_per_cell),
                    perm_span[i], dim);
              }
            },
            nb::arg("x"), nb::arg("cell_permutations"), nb::arg("dim"))
        .def(
            "T_apply",
            [](const dolfinx::fem::FiniteElement<T>& self,
               nb::ndarray<std::complex<T>, nb::ndim<1>, nb::c_contig> x,
               nb::ndarray<const std::uint32_t, nb::ndim<1>, nb::c_contig>
                   cell_permutations,
               int dim)
            {
              const std::size_t data_per_cell
                  = x.size() / cell_permutations.size();
              std::span<std::complex<T>> x_span(x.data(), x.size());
              std::span<const std::uint32_t> perm_span(
                  cell_permutations.data(), cell_permutations.size());

              for (std::size_t i = 0; i < cell_permutations.size(); i++)
              {
                self.T_apply(x_span.subspan(i * data_per_cell, data_per_cell),
                             perm_span[i], dim);
              }
            },
            nb::arg("x"), nb::arg("cell_permutations"), nb::arg("dim"))
        .def(
            "Tt_apply",
            [](const dolfinx::fem::FiniteElement<T>& self,
               nb::ndarray<std::complex<T>, nb::ndim<1>, nb::c_contig> x,
               nb::ndarray<const std::uint32_t, nb::ndim<1>, nb::c_contig>
                   cell_permutations,
               int dim)
            {
              const std::size_t data_per_cell
                  = x.size() / cell_permutations.size();
              std::span<std::complex<T>> x_span(x.data(), x.size());
              std::span<const std::uint32_t> perm_span(
                  cell_permutations.data(), cell_permutations.size());

              for (std::size_t i = 0; i < cell_permutations.size(); i++)
              {
                self.Tt_apply(x_span.subspan(i * data_per_cell, data_per_cell),
                              perm_span[i], dim);
              }
            },
            nb::arg("x"), nb::arg("cell_permutations"), nb::arg("dim"))
        .def(
            "Tt_inv_apply",
            [](const dolfinx::fem::FiniteElement<T>& self,
               nb::ndarray<std::complex<T>, nb::ndim<1>, nb::c_contig> x,
               nb::ndarray<const std::uint32_t, nb::ndim<1>, nb::c_contig>
                   cell_permutations,
               int dim)
            {
              const std::size_t data_per_cell
                  = x.size() / cell_permutations.size();
              std::span<std::complex<T>> x_span(x.data(), x.size());
              std::span<const std::uint32_t> perm_span(
                  cell_permutations.data(), cell_permutations.size());

              for (std::size_t i = 0; i < cell_permutations.size(); i++)
              {
                self.Tt_inv_apply(
                    x_span.subspan(i * data_per_cell, data_per_cell),
                    perm_span[i], dim);
              }
            },
            nb::arg("x"), nb::arg("cell_permutations"), nb::arg("dim"))
        .def_prop_ro("needs_dof_transformations",
                     &dolfinx::fem::FiniteElement<T>::needs_dof_transformations)
        .def("signature", &dolfinx::fem::FiniteElement<T>::signature);
  }
}

// Declare DirichletBC objects for type T
template <typename T>
void declare_objects(nb::module_& m, const std::string& type)
{
  using U = typename dolfinx::scalar_value_type_t<T>;

  // dolfinx::fem::DirichletBC
  std::string pyclass_name = std::string("DirichletBC_") + type;
  nb::class_<dolfinx::fem::DirichletBC<T, U>> dirichletbc(
      m, pyclass_name.c_str(),
      "Object for representing Dirichlet (essential) boundary "
      "conditions");

  dirichletbc
      .def(
          "__init__",
          [](dolfinx::fem::DirichletBC<T, U>* bc,
             nb::ndarray<const T, nb::c_contig> g,
             nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> dofs,
             std::shared_ptr<const dolfinx::fem::FunctionSpace<U>> V)
          {
            std::vector<std::size_t> shape(g.shape_ptr(),
                                           g.shape_ptr() + g.ndim());
            auto _g = std::make_shared<dolfinx::fem::Constant<T>>(
                std::span(g.data(), g.size()), shape);
            new (bc) dolfinx::fem::DirichletBC<T, U>(
                _g, std::vector(dofs.data(), dofs.data() + dofs.size()), V);
          },
          nb::arg("g").noconvert(), nb::arg("dofs").noconvert(), nb::arg("V"))
      .def(
          "__init__",
          [](dolfinx::fem::DirichletBC<T, U>* bc,
             std::shared_ptr<const dolfinx::fem::Constant<T>> g,
             nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> dofs,
             std::shared_ptr<const dolfinx::fem::FunctionSpace<U>> V)
          {
            new (bc) dolfinx::fem::DirichletBC<T, U>(
                g, std::vector(dofs.data(), dofs.data() + dofs.size()), V);
          },
          nb::arg("g").noconvert(), nb::arg("dofs").noconvert(), nb::arg("V"))
      .def(
          "__init__",
          [](dolfinx::fem::DirichletBC<T, U>* bc,
             std::shared_ptr<const dolfinx::fem::Function<T, U>> g,
             nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> dofs)
          {
            new (bc) dolfinx::fem::DirichletBC<T, U>(
                g, std::vector(dofs.data(), dofs.data() + dofs.size()));
          },
          nb::arg("g").noconvert(), nb::arg("dofs"))
      .def(
          "__init__",
          [](dolfinx::fem::DirichletBC<T, U>* bc,
             std::shared_ptr<const dolfinx::fem::Function<T, U>> g,
             std::array<
                 nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig>, 2>
                 V_g_dofs,
             std::shared_ptr<const dolfinx::fem::FunctionSpace<U>> V)
          {
            std::array dofs
                = {std::vector(V_g_dofs[0].data(),
                               V_g_dofs[0].data() + V_g_dofs[0].size()),
                   std::vector(V_g_dofs[1].data(),
                               V_g_dofs[1].data() + V_g_dofs[1].size())};
            new (bc) dolfinx::fem::DirichletBC(g, std::move(dofs), V);
          },
          nb::arg("g").noconvert(), nb::arg("dofs").noconvert(),
          nb::arg("V").noconvert())
      .def_prop_ro("dtype", [](const dolfinx::fem::Function<T, U>&)
                   { return dolfinx_wrappers::numpy_dtype<T>(); })
      .def("dof_indices",
           [](const dolfinx::fem::DirichletBC<T, U>& self)
           {
             auto [dofs, owned] = self.dof_indices();
             return std::pair(nb::ndarray<const std::int32_t, nb::numpy>(
                                  dofs.data(), {dofs.size()}, nb::handle()),
                              owned);
           })
      .def_prop_ro("function_space",
                   &dolfinx::fem::DirichletBC<T, U>::function_space)
      .def_prop_ro("value", &dolfinx::fem::DirichletBC<T, U>::value);

  // dolfinx::fem::Function
  std::string pyclass_name_function = std::string("Function_") + type;
  nb::class_<dolfinx::fem::Function<T, U>>(m, pyclass_name_function.c_str(),
                                           "A finite element function")
      .def(nb::init<std::shared_ptr<const dolfinx::fem::FunctionSpace<U>>>(),
           "Create a function on the given function space")
      .def(nb::init<std::shared_ptr<dolfinx::fem::FunctionSpace<U>>,
                    std::shared_ptr<dolfinx::la::Vector<T>>>())
      .def_rw("name", &dolfinx::fem::Function<T, U>::name)
      .def("sub", &dolfinx::fem::Function<T, U>::sub,
           "Return sub-function (view into parent Function")
      .def("collapse", &dolfinx::fem::Function<T, U>::collapse,
           "Collapse sub-function view")
      .def(
          "interpolate",
          [](dolfinx::fem::Function<T, U>& self,
             nb::ndarray<const T, nb::ndim<1>, nb::c_contig> f,
             nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> cells)
          {
            dolfinx::fem::interpolate(self, std::span(f.data(), f.size()),
                                      {1, f.size()},
                                      std::span(cells.data(), cells.size()));
          },
          nb::arg("f"), nb::arg("cells"), "Interpolate an expression function")
      .def(
          "interpolate",
          [](dolfinx::fem::Function<T, U>& self,
             nb::ndarray<const T, nb::ndim<2>, nb::c_contig> f,
             nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> cells)
          {
            dolfinx::fem::interpolate(self, std::span(f.data(), f.size()),
                                      {f.shape(0), f.shape(1)},
                                      std::span(cells.data(), cells.size()));
          },
          nb::arg("f"), nb::arg("cells"), "Interpolate an expression function")
      .def(
          "interpolate",
          [](dolfinx::fem::Function<T, U>& self,
             dolfinx::fem::Function<T, U>& u0,
             nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> cells0,
             nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> cells1)
          {
            self.interpolate(u0, std::span(cells0.data(), cells0.size()),
                             std::span(cells1.data(), cells1.size()));
          },
          nb::arg("u"), nb::arg("cells0"), nb::arg("cells1"),
          "Interpolate a finite element function")
      .def(
          "interpolate",
          [](dolfinx::fem::Function<T, U>& self,
             dolfinx::fem::Function<T, U>& u,
             nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> cells,
             const dolfinx::geometry::PointOwnershipData<U>& interpolation_data)
          {
            self.interpolate(u, std::span(cells.data(), cells.size()),
                             interpolation_data);
          },
          nb::arg("u"), nb::arg("cells"), nb::arg("interpolation_data"),
          "Interpolate a finite element function on non-matching meshes")
      .def(
          "interpolate_ptr",
          [](dolfinx::fem::Function<T, U>& self, std::uintptr_t addr,
             nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> cells)
          {
            assert(self.function_space());
            auto element = self.function_space()->element();
            assert(element);

            assert(self.function_space()->mesh());
            const std::vector<U> x = dolfinx::fem::interpolation_coords(
                *element, self.function_space()->mesh()->geometry(),
                std::span(cells.data(), cells.size()));

            const int gdim = self.function_space()->mesh()->geometry().dim();

            // Compute value size
            auto vshape = self.function_space()->value_shape();
            std::size_t value_size = std::reduce(vshape.begin(), vshape.end(),
                                                 1, std::multiplies{});

            std::array<std::size_t, 2> shape{value_size, x.size() / 3};
            std::vector<T> values(shape[0] * shape[1]);
            std::function<void(T*, int, int, const U*)> f
                = reinterpret_cast<void (*)(T*, int, int, const U*)>(addr);
            f(values.data(), shape[1], shape[0], x.data());
            dolfinx::fem::interpolate(self, std::span<const T>(values), shape,
                                      std::span(cells.data(), cells.size()));
          },
          nb::arg("f_ptr"), nb::arg("cells"),
          "Interpolate using a pointer to an expression with a C signature")
      .def(
          "interpolate",
          [](dolfinx::fem::Function<T, U>& self,
             const dolfinx::fem::Expression<T, U>& e0,
             nb::ndarray<const std::int32_t, nb::c_contig> cells0,
             nb::ndarray<const std::int32_t, nb::c_contig> cells1)
          {
            self.interpolate(e0, std::span(cells0.data(), cells0.size()),
                             std::span(cells1.data(), cells1.size()));
          },
          nb::arg("e0"), nb::arg("cells0"), nb::arg("cells1"),
          "Interpolate an Expression on a set of cells")
      .def_prop_ro(
          "x", nb::overload_cast<>(&dolfinx::fem::Function<T, U>::x),
          "Return the vector associated with the finite element Function")
      .def(
          "eval",
          [](const dolfinx::fem::Function<T, U>& self,
             nb::ndarray<const U, nb::ndim<2>, nb::c_contig> x,
             nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> cells,
             nb::ndarray<T, nb::ndim<2>, nb::c_contig> u)
          {
            // TODO: handle 1d case
            self.eval(std::span(x.data(), x.size()), {x.shape(0), x.shape(1)},
                      std::span(cells.data(), cells.size()),
                      std::span<T>(u.data(), u.size()),
                      {u.shape(0), u.shape(1)});
          },
          nb::arg("x"), nb::arg("cells"), nb::arg("values"),
          "Evaluate Function")
      .def_prop_ro("function_space",
                   &dolfinx::fem::Function<T, U>::function_space);

  // dolfinx::fem::Constant
  std::string pyclass_name_constant = std::string("Constant_") + type;
  nb::class_<dolfinx::fem::Constant<T>>(
      m, pyclass_name_constant.c_str(),
      "Value constant with respect to integration domain")
      .def(
          "__init__",
          [](dolfinx::fem::Constant<T>* cp,
             nb::ndarray<const T, nb::c_contig> c)
          {
            std::vector<std::size_t> shape(c.shape_ptr(),
                                           c.shape_ptr() + c.ndim());
            new (cp)
                dolfinx::fem::Constant<T>(std::span(c.data(), c.size()), shape);
          },
          nb::arg("c").noconvert(), "Create a constant from a value array")
      .def_prop_ro("dtype", [](const dolfinx::fem::Constant<T>)
                   { return dolfinx_wrappers::numpy_dtype<T>(); })
      .def_prop_ro(
          "value",
          [](dolfinx::fem::Constant<T>& self)
          {
            return nb::ndarray<T, nb::numpy>(self.value.data(),
                                             self.shape.size(),
                                             self.shape.data(), nb::handle());
          },
          nb::rv_policy::reference_internal);

  // dolfinx::fem::Expression
  std::string pyclass_name_expr = std::string("Expression_") + type;
  nb::class_<dolfinx::fem::Expression<T, U>>(m, pyclass_name_expr.c_str(),
                                             "An Expression")
      .def(
          "__init__",
          [](dolfinx::fem::Expression<T, U>* ex,
             const std::vector<std::shared_ptr<
                 const dolfinx::fem::Function<T, U>>>& coefficients,
             const std::vector<
                 std::shared_ptr<const dolfinx::fem::Constant<T>>>& constants,
             nb::ndarray<const U, nb::ndim<2>, nb::c_contig> X,
             std::uintptr_t fn_addr, const std::vector<int>& value_shape,
             std::shared_ptr<const dolfinx::fem::FunctionSpace<U>>
                 argument_function_space)
          {
            auto tabulate_expression_ptr
                = (void (*)(T*, const T*, const T*,
                            const typename geom_type<T>::value_type*,
                            const int*, const std::uint8_t*))fn_addr;
            new (ex) dolfinx::fem::Expression<T, U>(
                coefficients, constants, std::span(X.data(), X.size()),
                {X.shape(0), X.shape(1)}, tabulate_expression_ptr, value_shape,
                argument_function_space);
          },
          nb::arg("coefficients"), nb::arg("constants"), nb::arg("X"),
          nb::arg("fn"), nb::arg("value_shape"),
          nb::arg("argument_function_space"))
      .def(
          "eval",
          [](const dolfinx::fem::Expression<T, U>& self,
             const dolfinx::mesh::Mesh<U>& mesh,
             nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> cells,
             nb::ndarray<T, nb::ndim<2>, nb::c_contig> values)
          {
            std::span<T> foo(values.data(), values.size());
            self.eval(mesh, std::span(cells.data(), cells.size()), foo,
                      {values.shape(0), values.shape(1)});
          },
          nb::arg("mesh"), nb::arg("active_cells"), nb::arg("values"))
      .def("X",
           [](const dolfinx::fem::Expression<T, U>& self)
           {
             auto [X, shape] = self.X();
             return dolfinx_wrappers::as_nbarray(std::move(X), shape.size(),
                                                 shape.data());
           })
      .def_prop_ro("dtype", [](const dolfinx::fem::Expression<T, U>&)
                   { return dolfinx_wrappers::numpy_dtype<T>(); })
      .def_prop_ro("value_size", &dolfinx::fem::Expression<T, U>::value_size)
      .def_prop_ro("value_shape", &dolfinx::fem::Expression<T, U>::value_shape);

  std::string pymethod_create_expression
      = std::string("create_expression_") + type;
  m.def(
      pymethod_create_expression.c_str(),
      [](std::uintptr_t expression,
         const std::vector<std::shared_ptr<const dolfinx::fem::Function<T, U>>>&
             coefficients,
         const std::vector<std::shared_ptr<const dolfinx::fem::Constant<T>>>&
             constants,
         std::shared_ptr<const dolfinx::fem::FunctionSpace<U>>
             argument_function_space)
      {
        const ufcx_expression* p
            = reinterpret_cast<const ufcx_expression*>(expression);
        return dolfinx::fem::create_expression<T, U>(
            *p, coefficients, constants, argument_function_space);
      },
      nb::arg("expression"), nb::arg("coefficients"), nb::arg("constants"),
      nb::arg("argument_function_space").none(),
      "Create Expression from a pointer to ufc_form.");
}

template <typename T>
void declare_form(nb::module_& m, std::string type)
{
  using U = typename dolfinx::scalar_value_type_t<T>;

  // dolfinx::fem::Form
  std::string pyclass_name_form = std::string("Form_") + type;
  nb::class_<dolfinx::fem::Form<T, U>>(m, pyclass_name_form.c_str(),
                                       "Variational form object")
      .def(
          "__init__",
          [](dolfinx::fem::Form<T, U>* fp,
             const std::vector<
                 std::shared_ptr<const dolfinx::fem::FunctionSpace<U>>>& spaces,
             const std::map<
                 dolfinx::fem::IntegralType,
                 std::vector<std::tuple<
                     int, std::uintptr_t,
                     nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig>,
                     nb::ndarray<const std::int8_t, nb::ndim<1>,
                                 nb::c_contig>>>>& integrals,
             const std::vector<std::shared_ptr<
                 const dolfinx::fem::Function<T, U>>>& coefficients,
             const std::vector<
                 std::shared_ptr<const dolfinx::fem::Constant<T>>>& constants,
             bool needs_permutation_data,
             const std::map<std::shared_ptr<const dolfinx::mesh::Mesh<U>>,
                            nb::ndarray<const std::int32_t, nb::ndim<1>,
                                        nb::c_contig>>& entity_maps,
             std::shared_ptr<const dolfinx::mesh::Mesh<U>> mesh)
          {
            std::map<dolfinx::fem::IntegralType,
                     std::vector<dolfinx::fem::integral_data<T>>>
                _integrals;

            // Loop over kernel for each entity type
            for (auto& [type, kernels] : integrals)
            {
              for (auto& [id, ptr, e, c] : kernels)
              {
                auto kn_ptr
                    = (void (*)(T*, const T*, const T*,
                                const typename geom_type<T>::value_type*,
                                const int*, const std::uint8_t*))ptr;
                _integrals[type].emplace_back(
                    id, kn_ptr,
                    std::span<const std::int32_t>(e.data(), e.size()),
                    std::vector<int>(c.data(), c.data() + c.size()));
              }
            }

            std::map<std::shared_ptr<const dolfinx::mesh::Mesh<U>>,
                     std::span<const int32_t>>
                _entity_maps;
            for (auto& [msh, map] : entity_maps)
              _entity_maps.emplace(msh, std::span(map.data(), map.size()));
            new (fp) dolfinx::fem::Form<T, U>(
                spaces, std::move(_integrals), coefficients, constants,
                needs_permutation_data, _entity_maps, mesh);
          },
          nb::arg("spaces"), nb::arg("integrals"), nb::arg("coefficients"),
          nb::arg("constants"), nb::arg("need_permutation_data"),
          nb::arg("entity_maps"), nb::arg("mesh").none())
      .def(
          "__init__",
          [](dolfinx::fem::Form<T, U>* fp, std::uintptr_t form,
             const std::vector<
                 std::shared_ptr<const dolfinx::fem::FunctionSpace<U>>>& spaces,
             const std::vector<std::shared_ptr<
                 const dolfinx::fem::Function<T, U>>>& coefficients,
             const std::vector<
                 std::shared_ptr<const dolfinx::fem::Constant<T>>>& constants,
             const std::map<
                 dolfinx::fem::IntegralType,
                 std::vector<std::pair<
                     std::int32_t, nb::ndarray<const std::int32_t, nb::ndim<1>,
                                               nb::c_contig>>>>& subdomains,
             const std::map<std::shared_ptr<const dolfinx::mesh::Mesh<U>>,
                            nb::ndarray<const std::int32_t, nb::ndim<1>,
                                        nb::c_contig>>& entity_maps,
             std::shared_ptr<const dolfinx::mesh::Mesh<U>> mesh)
          {
            std::map<dolfinx::fem::IntegralType,
                     std::vector<std::pair<std::int32_t,
                                           std::span<const std::int32_t>>>>
                sd;
            for (auto& [itg, data] : subdomains)
            {
              std::vector<
                  std::pair<std::int32_t, std::span<const std::int32_t>>>
                  x;
              for (auto& [id, e] : data)
                x.emplace_back(id, std::span(e.data(), e.size()));
              sd.insert({itg, std::move(x)});
            }

            std::map<std::shared_ptr<const dolfinx::mesh::Mesh<U>>,
                     std::span<const int32_t>>
                _entity_maps;
            for (auto& [msh, map] : entity_maps)
              _entity_maps.emplace(msh, std::span(map.data(), map.size()));
            ufcx_form* p = reinterpret_cast<ufcx_form*>(form);
            new (fp)
                dolfinx::fem::Form<T, U>(dolfinx::fem::create_form_factory<T>(
                    *p, spaces, coefficients, constants, sd, _entity_maps,
                    mesh));
          },
          nb::arg("form"), nb::arg("spaces"), nb::arg("coefficients"),
          nb::arg("constants"), nb::arg("subdomains"), nb::arg("entity_maps"),
          nb::arg("mesh").none(), "Create a Form from a pointer to a ufcx_form")
      .def_prop_ro("dtype", [](const dolfinx::fem::Form<T, U>&)
                   { return dolfinx_wrappers::numpy_dtype<T>(); })
      .def_prop_ro("coefficients", &dolfinx::fem::Form<T, U>::coefficients)
      .def_prop_ro("rank", &dolfinx::fem::Form<T, U>::rank)
      .def_prop_ro("mesh", &dolfinx::fem::Form<T, U>::mesh)
      .def_prop_ro("function_spaces",
                   &dolfinx::fem::Form<T, U>::function_spaces)
      .def(
          "integral_ids",
          [](const dolfinx::fem::Form<T, U>& self,
             dolfinx::fem::IntegralType type)
          {
            auto ids = self.integral_ids(type);
            return dolfinx_wrappers::as_nbarray(std::move(ids));
          },
          nb::arg("type"))
      .def_prop_ro("integral_types", &dolfinx::fem::Form<T, U>::integral_types)
      .def_prop_ro("needs_facet_permutations",
                   &dolfinx::fem::Form<T, U>::needs_facet_permutations)
      .def(
          "domains",
          [](const dolfinx::fem::Form<T, U>& self,
             dolfinx::fem::IntegralType type, int i)
          {
            std::span<const std::int32_t> _d = self.domain(type, i);
            switch (type)
            {
            case dolfinx::fem::IntegralType::cell:
              return nb::ndarray<const std::int32_t, nb::numpy>(
                  _d.data(), {_d.size()}, nb::handle());
            case dolfinx::fem::IntegralType::exterior_facet:
            {
              return nb::ndarray<const std::int32_t, nb::numpy>(
                  _d.data(), {_d.size() / 2, 2}, nb::handle());
            }
            case dolfinx::fem::IntegralType::interior_facet:
            {
              return nb::ndarray<const std::int32_t, nb::numpy>(
                  _d.data(), {_d.size() / 4, 2, 2}, nb::handle());
            }
            default:
              throw ::std::runtime_error("Integral type unsupported.");
            }
          },
          nb::rv_policy::reference_internal, nb::arg("type"), nb::arg("i"));

  // Form
  std::string pymethod_create_form = std::string("create_form_") + type;
  m.def(
      pymethod_create_form.c_str(),
      [](std::uintptr_t form,
         const std::vector<
             std::shared_ptr<const dolfinx::fem::FunctionSpace<U>>>& spaces,
         const std::vector<std::shared_ptr<const dolfinx::fem::Function<T, U>>>&
             coefficients,
         const std::vector<std::shared_ptr<const dolfinx::fem::Constant<T>>>&
             constants,
         const std::map<
             dolfinx::fem::IntegralType,
             std::vector<std::pair<
                 std::int32_t, nb::ndarray<const std::int32_t, nb::c_contig>>>>&
             subdomains,
         std::shared_ptr<const dolfinx::mesh::Mesh<U>> mesh)
      {
        std::map<
            dolfinx::fem::IntegralType,
            std::vector<std::pair<std::int32_t, std::span<const std::int32_t>>>>
            sd;
        for (auto& [itg, data] : subdomains)
        {
          std::vector<std::pair<std::int32_t, std::span<const std::int32_t>>> x;
          for (auto& [id, idx] : data)
            x.emplace_back(id, std::span(idx.data(), idx.size()));
          sd.insert({itg, std::move(x)});
        }

        ufcx_form* p = reinterpret_cast<ufcx_form*>(form);
        return dolfinx::fem::create_form_factory<T>(*p, spaces, coefficients,
                                                    constants, sd, {}, mesh);
      },
      nb::arg("form"), nb::arg("spaces"), nb::arg("coefficients"),
      nb::arg("constants"), nb::arg("subdomains"), nb::arg("mesh"),
      "Create Form from a pointer to ufcx_form.");
  m.def(
      pymethod_create_form.c_str(),
      [](std::uintptr_t form,
         const std::vector<
             std::shared_ptr<const dolfinx::fem::FunctionSpace<U>>>& spaces,
         const std::map<std::string,
                        std::shared_ptr<const dolfinx::fem::Function<T, U>>>&
             coefficients,
         const std::map<std::string,
                        std::shared_ptr<const dolfinx::fem::Constant<T>>>&
             constants,
         const std::map<
             dolfinx::fem::IntegralType,
             std::vector<std::pair<
                 std::int32_t, nb::ndarray<const std::int32_t, nb::c_contig>>>>&
             subdomains,
         const std::map<std::shared_ptr<const dolfinx::mesh::Mesh<U>>,
                        nb::ndarray<const std::int32_t, nb::c_contig>>&
             entity_maps,
         std::shared_ptr<const dolfinx::mesh::Mesh<U>> mesh = nullptr)
      {
        std::map<
            dolfinx::fem::IntegralType,
            std::vector<std::pair<std::int32_t, std::span<const std::int32_t>>>>
            sd;
        for (auto& [itg, data] : subdomains)
        {
          std::vector<std::pair<std::int32_t, std::span<const std::int32_t>>> x;
          for (auto& [id, idx] : data)
            x.emplace_back(id, std::span(idx.data(), idx.size()));
          sd.insert({itg, std::move(x)});
        }
        std::map<std::shared_ptr<const dolfinx::mesh::Mesh<U>>,
                 std::span<const int32_t>>
            _entity_maps;
        for (auto& [msh, map] : entity_maps)
          _entity_maps.emplace(msh, std::span(map.data(), map.size()));
        ufcx_form* p = reinterpret_cast<ufcx_form*>(form);
        return dolfinx::fem::create_form<T, U>(
            *p, spaces, coefficients, constants, sd, _entity_maps, mesh);
      },
      nb::arg("form"), nb::arg("spaces"), nb::arg("coefficients"),
      nb::arg("constants"), nb::arg("subdomains"), nb::arg("entity_maps"),
      nb::arg("mesh"), "Create Form from a pointer to ufcx_form.");

  m.def("create_sparsity_pattern",
        &dolfinx::fem ::create_sparsity_pattern<T, U>, nb::arg("a"),
        "Create a sparsity pattern.");
}

template <typename T>
void declare_cmap(nb::module_& m, std::string type)
{
  std::string pyclass_name = std::string("CoordinateElement_") + type;
  nb::class_<dolfinx::fem::CoordinateElement<T>>(m, pyclass_name.c_str(),
                                                 "Coordinate map element")
      .def(nb::init<dolfinx::mesh::CellType, int>(), nb::arg("celltype"),
           nb::arg("degree"))
      .def(nb::init<std::shared_ptr<const basix::FiniteElement<T>>>(),
           nb::arg("element"))
      .def(
          "__init__",
          [](dolfinx::fem::CoordinateElement<T>* cm, dolfinx::mesh::CellType ct,
             int d, int var)
          {
            new (cm) dolfinx::fem::CoordinateElement<T>(
                ct, d, static_cast<basix::element::lagrange_variant>(var));
          },
          nb::arg("celltype"), nb::arg("degree"), nb::arg("variant"))
      .def_prop_ro("dtype", [](const dolfinx::fem::CoordinateElement<T>&)
                   { return dolfinx_wrappers::numpy_dtype<T>(); })
      .def("create_dof_layout",
           &dolfinx::fem::CoordinateElement<T>::create_dof_layout)
      .def_prop_ro("degree", &dolfinx::fem::CoordinateElement<T>::degree)
      .def_prop_ro("dim", &dolfinx::fem::CoordinateElement<T>::dim)
      .def_prop_ro("variant", [](const dolfinx::fem::CoordinateElement<T>& self)
                   { return static_cast<int>(self.variant()); })
      .def(
          "push_forward",
          [](const dolfinx::fem::CoordinateElement<T>& self,
             nb::ndarray<const T, nb::ndim<2>, nb::c_contig> X,
             nb::ndarray<const T, nb::ndim<2>, nb::c_contig> cell_x)
          {
            using mdspan2_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
                T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>;
            using cmdspan2_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
                const T,
                MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>;
            using cmdspan4_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
                const T,
                MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 4>>;

            std::array<std::size_t, 2> Xshape{X.shape(0), X.shape(1)};
            std::array<std::size_t, 4> phi_shape
                = self.tabulate_shape(0, X.shape(0));
            std::vector<T> phi_b(std::reduce(phi_shape.begin(), phi_shape.end(),
                                             1, std::multiplies{}));
            cmdspan4_t phi_full(phi_b.data(), phi_shape);
            self.tabulate(0, std::span(X.data(), X.size()), Xshape, phi_b);
            auto phi = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                phi_full, 0, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
                MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 0);

            std::array<std::size_t, 2> shape = {X.shape(0), cell_x.shape(1)};
            std::vector<T> xb(shape[0] * shape[1]);
            self.push_forward(
                mdspan2_t(xb.data(), shape),
                cmdspan2_t(cell_x.data(), cell_x.shape(0), cell_x.shape(1)),
                phi);

            return dolfinx_wrappers::as_nbarray(std::move(xb),
                                                {X.shape(0), cell_x.shape(1)});
          },
          nb::arg("X"), nb::arg("cell_geometry"))
      .def(
          "pull_back",
          [](const dolfinx::fem::CoordinateElement<T>& self,
             nb::ndarray<const T, nb::ndim<2>, nb::c_contig> x,
             nb::ndarray<const T, nb::ndim<2>, nb::c_contig> cell_geometry)
          {
            std::size_t num_points = x.shape(0);
            std::size_t gdim = x.shape(1);
            std::size_t tdim = dolfinx::mesh::cell_dim(self.cell_shape());

            using mdspan2_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
                T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>;
            using cmdspan2_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
                const T,
                MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>;
            using cmdspan4_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
                const T,
                MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 4>>;

            std::vector<T> Xb(num_points * tdim);
            mdspan2_t X(Xb.data(), num_points, tdim);
            cmdspan2_t _x(x.data(), x.shape(0), x.shape(1));
            cmdspan2_t g(cell_geometry.data(), cell_geometry.shape(0),
                         cell_geometry.shape(1));

            if (self.is_affine())
            {
              std::vector<T> J_b(gdim * tdim);
              mdspan2_t J(J_b.data(), gdim, tdim);
              std::vector<T> K_b(tdim * gdim);
              mdspan2_t K(K_b.data(), tdim, gdim);

              std::array<std::size_t, 4> phi_shape = self.tabulate_shape(1, 1);
              std::vector<T> phi_b(std::reduce(
                  phi_shape.begin(), phi_shape.end(), 1, std::multiplies{}));
              cmdspan4_t phi(phi_b.data(), phi_shape);

              self.tabulate(1, std::vector<T>(tdim), {1, tdim}, phi_b);
              auto dphi = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                  phi, std::pair(1, tdim + 1), 0,
                  MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 0);

              self.compute_jacobian(dphi, g, J);
              self.compute_jacobian_inverse(J, K);
              std::array<T, 3> x0 = {0, 0, 0};
              for (std::size_t i = 0; i < g.extent(1); ++i)
                x0[i] += g(0, i);
              self.pull_back_affine(X, K, x0, _x);
            }
            else
              self.pull_back_nonaffine(X, _x, g);

            return dolfinx_wrappers::as_nbarray(std::move(Xb),
                                                {num_points, tdim});
          },
          nb::arg("x"), nb::arg("cell_geometry"));
}

template <typename T>
void declare_real_functions(nb::module_& m)
{
  m.def(
      "create_element_dof_layout",
      [](const dolfinx::fem::FiniteElement<T>& element,
         const std::vector<int>& parent_map)
      { return dolfinx::fem::create_element_dof_layout(element, parent_map); },
      nb::arg("element"), nb::arg("parent_map"),
      "Create ElementDofLayout object from a ufc dofmap.");
  m.def(
      "create_dofmap",
      [](const dolfinx_wrappers::MPICommWrapper comm,
         dolfinx::mesh::Topology& topology,
         const dolfinx::fem::FiniteElement<T>& element)
      {
        dolfinx::fem::ElementDofLayout layout
            = dolfinx::fem::create_element_dof_layout(element);

        std::function<void(std::span<std::int32_t>, std::uint32_t)> permute_inv
            = nullptr;
        if (element.needs_dof_permutations())
          permute_inv = element.dof_permutation_fn(true, true);
        return dolfinx::fem::create_dofmap(comm.get(), layout, topology,
                                           permute_inv, nullptr);
      },
      nb::arg("comm"), nb::arg("topology"), nb::arg("element"),
      "Create DofMap object from an element.");
  m.def(
      "create_dofmaps",
      [](const dolfinx_wrappers::MPICommWrapper comm,
         dolfinx::mesh::Topology& topology,
         std::vector<std::shared_ptr<const dolfinx::fem::FiniteElement<T>>>
             elements)
      {
        std::vector<dolfinx::fem::ElementDofLayout> layouts;
        int D = topology.dim();
        assert(elements.size() == topology.entity_types(D).size());
        for (std::size_t i = 0; i < elements.size(); ++i)
        {
          layouts.push_back(
              dolfinx::fem::create_element_dof_layout(*elements[i]));
        }

        return dolfinx::fem::create_dofmaps(comm.get(), layouts, topology,
                                            nullptr, nullptr);
      },
      nb::arg("comm"), nb::arg("topology"), nb::arg("elements"),
      "Create DofMap objects on a mixed topology mesh from pointers to "
      "FiniteElements.");

  m.def(
      "locate_dofs_topological",
      [](const std::vector<
             std::shared_ptr<const dolfinx::fem::FunctionSpace<T>>>& V,
         int dim,
         nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> entities,
         bool remote)
      {
        if (V.size() != 2)
          throw std::runtime_error("Expected two function spaces.");
        std::array<std::vector<std::int32_t>, 2> dofs
            = dolfinx::fem::locate_dofs_topological(
                *V[0].get()->mesh()->topology_mutable(),
                {*V[0].get()->dofmap(), *V[1].get()->dofmap()}, dim,
                std::span(entities.data(), entities.size()), remote);
        return std::array<nb::ndarray<std::int32_t, nb::numpy>, 2>(
            {dolfinx_wrappers::as_nbarray(std::move(dofs[0])),
             dolfinx_wrappers::as_nbarray(std::move(dofs[1]))});
      },
      nb::arg("V"), nb::arg("dim"), nb::arg("entities"),
      nb::arg("remote") = true);
  m.def(
      "locate_dofs_topological",
      [](const dolfinx::fem::FunctionSpace<T>& V, int dim,
         nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> entities,
         bool remote)
      {
        return dolfinx_wrappers::as_nbarray(
            dolfinx::fem::locate_dofs_topological(
                *V.mesh()->topology_mutable(), *V.dofmap(), dim,
                std::span(entities.data(), entities.size()), remote));
      },
      nb::arg("V"), nb::arg("dim"), nb::arg("entities"),
      nb::arg("remote") = true);
  m.def(
      "locate_dofs_geometrical",
      [](const std::vector<
             std::shared_ptr<const dolfinx::fem::FunctionSpace<T>>>& V,
         std::function<nb::ndarray<bool, nb::ndim<1>, nb::c_contig>(
             nb::ndarray<const T, nb::ndim<2>, nb::numpy>)>
             marker)
      {
        if (V.size() != 2)
          throw std::runtime_error("Expected two function spaces.");

        auto _marker = [&marker](auto x)
        {
          nb::ndarray<const T, nb::ndim<2>, nb::numpy> x_view(
              x.data_handle(), {x.extent(0), x.extent(1)}, nb::handle());
          auto marked = marker(x_view);
          return std::vector<std::int8_t>(marked.data(),
                                          marked.data() + marked.size());
        };

        std::array<std::vector<std::int32_t>, 2> dofs
            = dolfinx::fem::locate_dofs_geometrical<T>({*V[0], *V[1]}, _marker);
        return std::array<nb::ndarray<std::int32_t, nb::numpy>, 2>(
            {dolfinx_wrappers::as_nbarray(std::move(dofs[0])),
             dolfinx_wrappers::as_nbarray(std::move(dofs[1]))});
      },
      nb::arg("V"), nb::arg("marker"));
  m.def(
      "locate_dofs_geometrical",
      [](const dolfinx::fem::FunctionSpace<T>& V,
         std::function<nb::ndarray<bool, nb::ndim<1>, nb::c_contig>(
             nb::ndarray<const T, nb::ndim<2>, nb::numpy>)>
             marker)
      {
        auto _marker = [&marker](auto x)
        {
          nb::ndarray<const T, nb::ndim<2>, nb::numpy> x_view(
              x.data_handle(), {x.extent(0), x.extent(1)}, nb::handle());
          auto marked = marker(x_view);
          return std::vector<std::int8_t>(marked.data(),
                                          marked.data() + marked.size());
        };

        return dolfinx_wrappers::as_nbarray(
            dolfinx::fem::locate_dofs_geometrical(V, _marker));
      },
      nb::arg("V"), nb::arg("marker"));

  m.def(
      "interpolation_coords",
      [](const dolfinx::fem::FiniteElement<T>& e,
         const dolfinx::mesh::Geometry<T>& geometry,
         nb::ndarray<std::int32_t, nb::ndim<1>, nb::c_contig> cells)
      {
        std::vector<T> x = dolfinx::fem::interpolation_coords(
            e, geometry, std::span(cells.data(), cells.size()));
        return dolfinx_wrappers::as_nbarray(std::move(x), {3, x.size() / 3});
      },
      nb::arg("element"), nb::arg("V"), nb::arg("cells"));

  m.def(
      "create_interpolation_data",
      [](const dolfinx::mesh::Geometry<T>& geometry0,
         const dolfinx::fem::FiniteElement<T>& element0,
         const dolfinx::mesh::Mesh<T>& mesh1,
         nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> cells,
         T padding)
      {
        return dolfinx::fem::create_interpolation_data(
            geometry0, element0, mesh1, std::span(cells.data(), cells.size()),
            padding);
      },
      nb::arg("geometry0"), nb::arg("element0"), nb::arg("mesh1"),
      nb::arg("cells"), nb ::arg("padding"));
}

} // namespace

namespace dolfinx_wrappers
{

void fem(nb::module_& m)
{
  declare_objects<float>(m, "float32");
  declare_objects<double>(m, "float64");
  declare_objects<std::complex<float>>(m, "complex64");
  declare_objects<std::complex<double>>(m, "complex128");

  declare_form<float>(m, "float32");
  declare_form<double>(m, "float64");
  declare_form<std::complex<float>>(m, "complex64");
  declare_form<std::complex<double>>(m, "complex128");

  // fem::CoordinateElement
  declare_cmap<float>(m, "float32");
  declare_cmap<double>(m, "float64");

  m.def(
      "build_dofmap",
      [](MPICommWrapper comm, const dolfinx::mesh::Topology& topology,
         const dolfinx::fem::ElementDofLayout& layout)
      {
        assert(topology.entity_types(topology.dim()).size() == 1);
        auto [map, bs, dofmap] = dolfinx::fem::build_dofmap_data(
            comm.get(), topology, {layout},
            [](const dolfinx::graph::AdjacencyList<std::int32_t>& g)
            { return dolfinx::graph::reorder_gps(g); });
        return std::tuple(std::move(map), bs, std::move(dofmap));
      },
      nb::arg("comm"), nb::arg("topology"), nb::arg("layout"),
      "Build a dofmap on a mesh.");
  m.def(
      "transpose_dofmap",
      [](nb::ndarray<const std::int32_t, nb::ndim<2>, nb::c_contig> dofmap,
         int num_cells)
      {
        MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
            const std::int32_t,
            MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
            _dofmap(dofmap.data(), dofmap.shape(0), dofmap.shape(1));
        return dolfinx::fem::transpose_dofmap(_dofmap, num_cells);
      },
      "Build the index to (cell, local index) map from a dofmap ((cell, local "
      "index) -> index).");
  m.def(
      "compute_integration_domains",
      [](dolfinx::fem::IntegralType type,
         const dolfinx::mesh::Topology& topology,
         const nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig>
             entities,
         int dim)
      {
        auto integration_entities = dolfinx::fem::compute_integration_domains(
            type, topology, std::span(entities.data(), entities.size()), dim);
        return dolfinx_wrappers::as_nbarray(std::move(integration_entities));
      },
      nb::arg("integral_type"), nb::arg("topology"), nb::arg("entities"),
      nb::arg("dim"));

  // dolfinx::fem::ElementDofLayout
  nb::class_<dolfinx::fem::ElementDofLayout>(
      m, "ElementDofLayout", "Object describing the layout of dofs on a cell")
      .def(nb::init<int, const std::vector<std::vector<std::vector<int>>>&,
                    const std::vector<std::vector<std::vector<int>>>&,
                    const std::vector<int>&,
                    const std::vector<dolfinx::fem::ElementDofLayout>&>(),
           nb::arg("block_size"), nb::arg("endity_dofs"),
           nb::arg("entity_closure_dofs"), nb::arg("parent_map"),
           nb::arg("sub_layouts"))
      .def_prop_ro("num_dofs", &dolfinx::fem::ElementDofLayout::num_dofs)
      .def("num_entity_dofs", &dolfinx::fem::ElementDofLayout::num_entity_dofs,
           nb::arg("dim"))
      .def("num_entity_closure_dofs",
           &dolfinx::fem::ElementDofLayout::num_entity_closure_dofs,
           nb::arg("dim"))
      .def("entity_dofs", &dolfinx::fem::ElementDofLayout::entity_dofs,
           nb::arg("dim"), nb::arg("entity_index"))
      .def("entity_closure_dofs",
           &dolfinx::fem::ElementDofLayout::entity_closure_dofs, nb::arg("dim"),
           nb::arg("entity_index"))
      .def_prop_ro("block_size", &dolfinx::fem::ElementDofLayout::block_size);

  // dolfinx::fem::DofMap
  nb::class_<dolfinx::fem::DofMap>(m, "DofMap", "DofMap object")
      .def(
          "__init__",
          [](dolfinx::fem::DofMap* self,
             const dolfinx::fem::ElementDofLayout& element,
             std::shared_ptr<const dolfinx::common::IndexMap> index_map,
             int index_map_bs,
             const dolfinx::graph::AdjacencyList<std::int32_t>& dofmap, int bs)
          {
            new (self) dolfinx::fem::DofMap(element, index_map, index_map_bs,
                                            dofmap.array(), bs);
          },
          nb::arg("element_dof_layout"), nb::arg("index_map"),
          nb::arg("index_map_bs"), nb::arg("dofmap"), nb::arg("bs"))
      .def_ro("index_map", &dolfinx::fem::DofMap::index_map)
      .def_prop_ro("index_map_bs", &dolfinx::fem::DofMap::index_map_bs)
      .def_prop_ro("dof_layout", &dolfinx::fem::DofMap::element_dof_layout)
      .def(
          "cell_dofs",
          [](const dolfinx::fem::DofMap& self, int cell)
          {
            std::span<const std::int32_t> dofs = self.cell_dofs(cell);
            return nb::ndarray<const std::int32_t, nb::numpy>(
                dofs.data(), {dofs.size()}, nb::handle());
          },
          nb::rv_policy::reference_internal, nb::arg("cell"))
      .def_prop_ro("bs", &dolfinx::fem::DofMap::bs)
      .def(
          "map",
          [](const dolfinx::fem::DofMap& self)
          {
            auto dofs = self.map();
            return nb::ndarray<const std::int32_t, nb::numpy>(
                dofs.data_handle(), {dofs.extent(0), dofs.extent(1)},
                nb::handle());
          },
          nb::rv_policy::reference_internal);

  nb::enum_<dolfinx::fem::IntegralType>(m, "IntegralType")
      .value("cell", dolfinx::fem::IntegralType::cell, "cell integral")
      .value("exterior_facet", dolfinx::fem::IntegralType::exterior_facet,
             "exterior facet integral")
      .value("interior_facet", dolfinx::fem::IntegralType::interior_facet,
             "exterior facet integral")
      .value("vertex", dolfinx::fem::IntegralType::vertex, "vertex integral");

  declare_function_space<float>(m, "float32");
  declare_function_space<double>(m, "float64");

  declare_real_functions<float>(m);
  declare_real_functions<double>(m);
}
} // namespace dolfinx_wrappers
