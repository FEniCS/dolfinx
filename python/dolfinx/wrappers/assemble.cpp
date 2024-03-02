// Copyright (C) 2017-2021 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "array.h"
#include "pycoeff.h"
#include <array>
#include <complex>
#include <cstdint>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/fem/assembler.h>
#include <dolfinx/fem/discreteoperators.h>
#include <dolfinx/fem/sparsitybuild.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/la/MatrixCSR.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/mesh/Mesh.h>
#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/operators.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <span>
#include <string>
#include <utility>

namespace nb = nanobind;

namespace
{

// Declare assembler function that have multiple scalar types
template <typename T, typename U>
void declare_discrete_operators(nb::module_& m)
{
  m.def(
      "discrete_gradient",
      [](const dolfinx::fem::FunctionSpace<U>& V0,
         const dolfinx::fem::FunctionSpace<U>& V1)
      {
        assert(V0.mesh());
        auto mesh = V0.mesh();
        assert(V1.mesh());
        assert(mesh == V1.mesh());
        MPI_Comm comm = mesh->comm();

        auto dofmap0 = V0.dofmap();
        assert(dofmap0);
        auto dofmap1 = V1.dofmap();
        assert(dofmap1);

        // Create and build  sparsity pattern
        assert(dofmap0->index_map);
        assert(dofmap1->index_map);
        dolfinx::la::SparsityPattern sp(
            comm, {dofmap1->index_map, dofmap0->index_map},
            {dofmap1->index_map_bs(), dofmap0->index_map_bs()});

        int tdim = mesh->topology()->dim();
        auto map = mesh->topology()->index_map(tdim);
        assert(map);
        std::vector<std::int32_t> c(map->size_local(), 0);
        std::iota(c.begin(), c.end(), 0);
        dolfinx::fem::sparsitybuild::cells(sp, {c, c}, {*dofmap1, *dofmap0});
        sp.finalize();

        // Build operator
        dolfinx::la::MatrixCSR<T> A(sp);
        dolfinx::fem::discrete_gradient<T, U>(
            *V0.mesh()->topology_mutable(), {*V0.element(), *V0.dofmap()},
            {*V1.element(), *V1.dofmap()}, A.mat_set_values());
        return A;
      },
      nb::arg("V0"), nb::arg("V1"));
}

// Declare assembler function that have multiple scalar types
template <typename T, typename U>
void declare_assembly_functions(nb::module_& m)
{
  // Coefficient/constant packing
  m.def(
      "pack_coefficients",
      [](const dolfinx::fem::Form<T, U>& form)
      {
        using Key_t = typename std::pair<dolfinx::fem::IntegralType, int>;

        // Pack coefficients
        std::map<Key_t, std::pair<std::vector<T>, int>> coeffs
            = dolfinx::fem::allocate_coefficient_storage(form);
        dolfinx::fem::pack_coefficients(form, coeffs);

        // Move into NumPy data structures
        std::map<Key_t, nb::ndarray<T, nb::numpy>> c;
        std::transform(
            coeffs.begin(), coeffs.end(), std::inserter(c, c.end()),
            [](auto& e) -> typename decltype(c)::value_type
            {
              std::size_t num_ents
                  = e.second.first.empty()
                        ? 0
                        : e.second.first.size() / e.second.second;
              return std::pair<const std::pair<dolfinx::fem::IntegralType, int>,
                               nb::ndarray<T, nb::numpy>>(
                  e.first,
                  dolfinx_wrappers::as_nbarray(
                      std::move(e.second.first),
                      {num_ents, static_cast<std::size_t>(e.second.second)}));
            });

        return c;
      },
      nb::arg("form"), "Pack coefficients for a Form.");
  m.def(
      "pack_constants",
      [](const dolfinx::fem::Form<T, U>& form) {
        return dolfinx_wrappers::as_nbarray(dolfinx::fem::pack_constants(form));
      },
      nb::arg("form"), "Pack constants for a Form.");
  m.def(
      "pack_constants",
      [](const dolfinx::fem::Expression<T, U>& e)
      { return dolfinx_wrappers::as_nbarray(dolfinx::fem::pack_constants(e)); },
      nb::arg("e"), "Pack constants for an Expression.");

  // Functional
  m.def(
      "assemble_scalar",
      [](const dolfinx::fem::Form<T, U>& M,
         nb::ndarray<const T, nb::ndim<1>, nb::c_contig> constants,
         const std::map<std::pair<dolfinx::fem::IntegralType, int>,
                        nb::ndarray<const T, nb::ndim<2>, nb::c_contig>>&
             coefficients)
      {
        return dolfinx::fem::assemble_scalar<T>(
            M, std::span(constants.data(), constants.size()),
            py_to_cpp_coeffs(coefficients));
      },
      nb::arg("M"), nb::arg("constants"), nb::arg("coefficients"),
      "Assemble functional over mesh with provided constants and "
      "coefficients");
  // Vector
  m.def(
      "assemble_vector",
      [](nb::ndarray<T, nb::ndim<1>, nb::c_contig> b,
         const dolfinx::fem::Form<T, U>& L,
         nb::ndarray<const T, nb::ndim<1>, nb::c_contig> constants,
         const std::map<std::pair<dolfinx::fem::IntegralType, int>,
                        nb::ndarray<const T, nb::ndim<2>, nb::c_contig>>&
             coefficients)
      {
        dolfinx::fem::assemble_vector<T>(
            std::span(b.data(), b.size()), L,
            std::span(constants.data(), constants.size()),
            py_to_cpp_coeffs(coefficients));
      },
      nb::arg("b"), nb::arg("L"), nb::arg("constants"), nb::arg("coeffs"),
      "Assemble linear form into an existing vector with pre-packed constants "
      "and coefficients");
  // MatrixCSR
  m.def(
      "assemble_matrix",
      [](dolfinx::la::MatrixCSR<T>& A, const dolfinx::fem::Form<T, U>& a,
         nb::ndarray<const T, nb::ndim<1>, nb::c_contig> constants,
         const std::map<std::pair<dolfinx::fem::IntegralType, int>,
                        nb::ndarray<const T, nb::ndim<2>, nb::c_contig>>&
             coefficients,
         const std::vector<
             std::shared_ptr<const dolfinx::fem::DirichletBC<T, U>>>& bcs)
      {
        const std::array<int, 2> data_bs
            = {a.function_spaces().at(0)->dofmap()->index_map_bs(),
               a.function_spaces().at(1)->dofmap()->index_map_bs()};

        if (data_bs[0] != data_bs[1])
          throw std::runtime_error(
              "Non-square blocksize unsupported in Python");

        if (data_bs[0] == 1)
        {
          dolfinx::fem::assemble_matrix(
              A.mat_add_values(), a,
              std::span<const T>(constants.data(), constants.size()),
              py_to_cpp_coeffs(coefficients), bcs);
        }
        else if (data_bs[0] == 2)
        {
          auto mat_add = A.template mat_add_values<2, 2>();
          dolfinx::fem::assemble_matrix(
              mat_add, a, std::span(constants.data(), constants.size()),
              py_to_cpp_coeffs(coefficients), bcs);
        }
        else if (data_bs[0] == 3)
        {
          auto mat_add = A.template mat_add_values<3, 3>();
          dolfinx::fem::assemble_matrix(
              mat_add, a, std::span(constants.data(), constants.size()),
              py_to_cpp_coeffs(coefficients), bcs);
        }
        else if (data_bs[0] == 4)
        {
          auto mat_add = A.template mat_add_values<4, 4>();
          dolfinx::fem::assemble_matrix(
              mat_add, a, std::span(constants.data(), constants.size()),
              py_to_cpp_coeffs(coefficients), bcs);
        }
        else if (data_bs[0] == 5)
        {
          auto mat_add = A.template mat_add_values<5, 5>();
          dolfinx::fem::assemble_matrix(
              mat_add, a, std::span(constants.data(), constants.size()),
              py_to_cpp_coeffs(coefficients), bcs);
        }
        else if (data_bs[0] == 6)
        {
          auto mat_add = A.template mat_add_values<6, 6>();
          dolfinx::fem::assemble_matrix(
              mat_add, a, std::span(constants.data(), constants.size()),
              py_to_cpp_coeffs(coefficients), bcs);
        }
        else if (data_bs[0] == 7)
        {
          auto mat_add = A.template mat_add_values<7, 7>();
          dolfinx::fem::assemble_matrix(
              mat_add, a, std::span(constants.data(), constants.size()),
              py_to_cpp_coeffs(coefficients), bcs);
        }
        else if (data_bs[0] == 8)
        {
          auto mat_add = A.template mat_add_values<8, 8>();
          dolfinx::fem::assemble_matrix(
              mat_add, a, std::span(constants.data(), constants.size()),
              py_to_cpp_coeffs(coefficients), bcs);
        }
        else if (data_bs[0] == 9)
        {
          auto mat_add = A.template mat_add_values<9, 9>();
          dolfinx::fem::assemble_matrix(
              mat_add, a, std::span(constants.data(), constants.size()),
              py_to_cpp_coeffs(coefficients), bcs);
        }
        else
          throw std::runtime_error("Block size not supported in Python");
      },
      nb::arg("A"), nb::arg("a"), nb::arg("constants"), nb::arg("coeffs"),
      nb::arg("bcs"), "Experimental.");
  m.def(
      "insert_diagonal",
      [](dolfinx::la::MatrixCSR<T>& A, const dolfinx::fem::FunctionSpace<U>& V,
         const std::vector<
             std::shared_ptr<const dolfinx::fem::DirichletBC<T, U>>>& bcs,
         T diagonal)
      {
        // NB block size of data ("diagonal") is (1, 1)
        dolfinx::fem::set_diagonal(A.mat_set_values(), V, bcs, diagonal);
      },
      nb::arg("A"), nb::arg("V"), nb::arg("bcs"), nb::arg("diagonal"),
      "Experimental.");
  m.def(
      "insert_diagonal",
      [](dolfinx::la::MatrixCSR<T>& A,
         nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> rows,
         T diagonal)
      {
        dolfinx::fem::set_diagonal(
            A.mat_set_values(), std::span(rows.data(), rows.size()), diagonal);
      },
      nb::arg("A"), nb::arg("rows"), nb::arg("diagonal"), "Experimental.");
  m.def(
      "assemble_matrix",
      [](std::function<int(
             nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig,
                         nb::numpy>,
             nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig,
                         nb::numpy>,
             nb::ndarray<const T, nb::ndim<2>, nb::c_contig, nb::numpy>)>
             fin,
         const dolfinx::fem::Form<T, U>& form,
         const std::vector<
             std::shared_ptr<const dolfinx::fem::DirichletBC<T, U>>>& bcs)
      {
        auto f = [&fin](std::span<const std::int32_t> rows,
                        std::span<const std::int32_t> cols,
                        std::span<const T> data)
        {
          return fin(nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig,
                                 nb::numpy>(rows.data(), {rows.size()}),
                     nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig,
                                 nb::numpy>(cols.data(), {cols.size()}),
                     nb::ndarray<const T, nb::ndim<2>, nb::c_contig, nb::numpy>(
                         data.data(), {data.size()}));
        };
        dolfinx::fem::assemble_matrix(f, form, bcs);
      },
      nb::arg("fin"), nb::arg("form"), nb::arg("bcs"),
      "Experimental assembly with Python insertion function. This will be "
      "slow. Use for testing only.");

  // BC modifiers
  m.def(
      "apply_lifting",
      [](nb::ndarray<T, nb::ndim<1>, nb::c_contig> b,
         const std::vector<std::shared_ptr<const dolfinx::fem::Form<T, U>>>& a,
         const std::vector<nb::ndarray<const T, nb::ndim<1>, nb::c_contig>>&
             constants,
         const std::vector<
             std::map<std::pair<dolfinx::fem::IntegralType, int>,
                      nb::ndarray<const T, nb::ndim<2>, nb::c_contig>>>& coeffs,
         const std::vector<std::vector<
             std::shared_ptr<const dolfinx::fem::DirichletBC<T, U>>>>& bcs1,
         const std::vector<nb::ndarray<const T, nb::ndim<1>, nb::c_contig>>& x0,
         T scale)
      {
        std::vector<std::span<const T>> _x0;
        for (auto x : x0)
          _x0.emplace_back(x.data(), x.size());

        std::vector<std::span<const T>> _constants;
        std::transform(
            constants.begin(), constants.end(), std::back_inserter(_constants),
            [](auto& c) { return std::span<const T>(c.data(), c.size()); });

        std::vector<std::map<std::pair<dolfinx::fem::IntegralType, int>,
                             std::pair<std::span<const T>, int>>>
            _coeffs;
        std::transform(coeffs.begin(), coeffs.end(),
                       std::back_inserter(_coeffs),
                       [](auto& c) { return py_to_cpp_coeffs(c); });

        dolfinx::fem::apply_lifting<T>(std::span<T>(b.data(), b.size()), a,
                                       _constants, _coeffs, bcs1, _x0, scale);
      },
      nb::arg("b").noconvert(), nb::arg("a"), nb::arg("constants"),
      nb::arg("coeffs"), nb::arg("bcs1"), nb::arg("x0"), nb::arg("scale"),
      "Modify vector for lifted boundary conditions");
  m.def(
      "set_bc",
      [](nb::ndarray<T, nb::ndim<1>, nb::c_contig> b,
         const std::vector<
             std::shared_ptr<const dolfinx::fem::DirichletBC<T, U>>>& bcs,
         nb::ndarray<const T, nb::ndim<1>, nb::c_contig> x0, T scale)
      {
        dolfinx::fem::set_bc<T>(std::span(b.data(), b.size()), bcs,
                                std::span(x0.data(), x0.shape(0)), scale);
      },
      nb::arg("b").noconvert(), nb::arg("bcs"), nb::arg("x0").noconvert(),
      nb::arg("scale"));
  m.def(
      "set_bc",
      [](nb::ndarray<T, nb::ndim<1>, nb::c_contig> b,
         const std::vector<
             std::shared_ptr<const dolfinx::fem::DirichletBC<T, U>>>& bcs,
         T scale)
      { dolfinx::fem::set_bc<T>(std::span(b.data(), b.size()), bcs, scale); },
      nb::arg("b").noconvert(), nb::arg("bcs"), nb::arg("scale"));
}

} // namespace

namespace dolfinx_wrappers
{

void assemble(nb::module_& m)
{
  // dolfinx::fem::assemble
  declare_assembly_functions<float, float>(m);
  declare_assembly_functions<double, double>(m);
  declare_assembly_functions<std::complex<float>, float>(m);
  declare_assembly_functions<std::complex<double>, double>(m);

  declare_discrete_operators<float, float>(m);
  declare_discrete_operators<double, double>(m);
  declare_discrete_operators<std::complex<float>, float>(m);
  declare_discrete_operators<std::complex<double>, double>(m);
}
} // namespace dolfinx_wrappers
