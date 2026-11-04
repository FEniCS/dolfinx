// Copyright (C) 2017-2026 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "array.h"
#include "pycoeff.h"
#include <array>
#include <basix/mdspan.hpp>
#include <cstddef>
#include <cstdint>
#include <dolfinx/common/types.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/fem/assembler.h>
#include <dolfinx/fem/discreteoperators.h>
#include <dolfinx/fem/expression_evaluate.h>
#include <dolfinx/fem/sparsitybuild.h>
#include <dolfinx/la/MatrixCSR.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/mesh/Mesh.h>
#include <functional>
#include <map>
#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/operators.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <numeric>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace dolfinx_wrappers
{
namespace nb = nanobind;

namespace md = MDSPAN_IMPL_STANDARD_NAMESPACE;

/// @brief Call `f` with `entities` viewed as a rank-1 mdspan of cells,
/// or a rank-2 mdspan of (cell, local entity index) pairs.
/// @throws std::invalid_argument if `entities` has another shape.
template <typename F>
decltype(auto)
visit_entities(nb::ndarray<const std::int32_t, nb::c_contig> entities, F&& f)
{
  if (entities.ndim() == 1)
    return f(md::mdspan(entities.data(), entities.shape(0)));
  else if (entities.ndim() == 2 and entities.shape(1) == 2)
  {
    return f(md::mdspan<const std::int32_t,
                        md::extents<std::size_t, md::dynamic_extent, 2>>(
        entities.data(), entities.shape(0), 2));
  }
  else
  {
    throw std::invalid_argument(
        "entities must have shape (num_entities,) or (num_entities, 2).");
  }
}

template <typename U>
dolfinx::la::SparsityPattern
create_sparsity(const dolfinx::fem::FunctionSpace<U>& V0,
                const dolfinx::fem::FunctionSpace<U>& V1)
{
  std::shared_ptr<const dolfinx::mesh::Mesh<U>> mesh = V0.mesh();
  if (!mesh or mesh != V1.mesh())
    throw std::invalid_argument("V0 and V1 must share a mesh.");
  MPI_Comm comm = mesh->comm();

  std::shared_ptr<const dolfinx::fem::DofMap> dofmap0 = V0.dofmap();
  std::shared_ptr<const dolfinx::fem::DofMap> dofmap1 = V1.dofmap();
  assert(dofmap0);
  assert(dofmap1);
  assert(dofmap0->index_map);
  assert(dofmap1->index_map);

  // Create and build sparsity pattern
  dolfinx::la::SparsityPattern sp(
      comm, {dofmap1->index_map, dofmap0->index_map},
      {dofmap1->index_map_bs(), dofmap0->index_map_bs()});

  int tdim = mesh->topology()->dim();
  auto map = mesh->topology()->index_map(tdim);
  assert(map);
  auto c = std::views::iota(0, map->size_local());
  dolfinx::fem::sparsitybuild::cells(sp, std::pair{c, c}, {*dofmap1, *dofmap0});
  sp.finalize();

  return sp;
}

/// Sparsity of a discrete derivative operator: the V1
/// degrees-of-freedom on each mesh entity against the V0
/// degrees-of-freedom on its closure.
template <typename U>
dolfinx::la::SparsityPattern
create_sparsity_entity_closure(const dolfinx::fem::FunctionSpace<U>& V0,
                               const dolfinx::fem::FunctionSpace<U>& V1)
{
  dolfinx::fem::impl::check_derivative_source(*V0.element());
  std::shared_ptr<const dolfinx::mesh::Mesh<U>> mesh = V0.mesh();
  if (!mesh or mesh != V1.mesh())
    throw std::invalid_argument("V0 and V1 must share a mesh.");

  std::shared_ptr<const dolfinx::fem::DofMap> dofmap0 = V0.dofmap();
  std::shared_ptr<const dolfinx::fem::DofMap> dofmap1 = V1.dofmap();
  assert(dofmap0);
  assert(dofmap1);
  dolfinx::la::SparsityPattern sp(
      mesh->comm(), {dofmap1->index_map, dofmap0->index_map},
      {dofmap1->index_map_bs(), dofmap0->index_map_bs()});

  int tdim = mesh->topology()->dim();
  auto map = mesh->topology()->index_map(tdim);
  assert(map);
  std::vector<std::int32_t> cells(map->size_local());
  std::iota(cells.begin(), cells.end(), 0);
  dolfinx::fem::sparsitybuild::entity_closure(sp, cells, {*dofmap1, *dofmap0});
  sp.finalize();

  return sp;
}

/// @brief Call `f(bs0, bs1)` with data block sizes `bs`, as
/// `std::integral_constant`s for the sizes with compile-time insertion
/// and as `int`s otherwise.
///
/// Compile-time sizes are (k, k) for k in {1, ..., 9}, and (1, k),
/// (k, 1) for k in {2, 3}.
template <typename F>
void visit_block_size(std::array<int, 2> bs, F&& f)
{
  static constexpr std::array<std::array<int, 2>, 13> sizes{{
      {1, 1},
      {2, 2},
      {3, 3},
      {4, 4},
      {5, 5},
      {6, 6},
      {7, 7},
      {8, 8},
      {9, 9},
      {1, 2},
      {2, 1},
      {1, 3},
      {3, 1},
  }};

  // Call f with sizes[I] as compile-time constants if it equals bs
  auto try_size
      = [&bs, &f]<std::size_t I>(std::integral_constant<std::size_t, I>) -> bool
  {
    if (bs != sizes[I])
      return false;

    f(std::integral_constant<int, sizes[I][0]>{},
      std::integral_constant<int, sizes[I][1]>{});
    return true;
  };

  // Try each entry of sizes in order, stopping at the first match
  bool found = [&try_size]<std::size_t... I>(std::index_sequence<I...>)
  {
    return (try_size(std::integral_constant<std::size_t, I>{}) or ...);
  }(std::make_index_sequence<sizes.size()>{});

  if (!found)
    f(bs[0], bs[1]);
}

// Declare assembler function that have multiple scalar types
template <typename T, typename U>
void declare_discrete_operators(nanobind::module_& m)
{
  m.def(
      "interpolation_matrix",
      [](const dolfinx::fem::FunctionSpace<U>& V0,
         const dolfinx::fem::FunctionSpace<U>& V1)
      {
        // Create sparsity
        dolfinx::la::SparsityPattern sp = create_sparsity(V0, V1);

        // Build operator
        dolfinx::la::MatrixCSR<T> A(sp);

        // Element data has the block sizes of V1 (rows) and V0 (columns)
        visit_block_size(
            {V1.element()->block_size(), V0.element()->block_size()},
            [&V0, &V1, &A](auto bs0, auto bs1)
            {
              dolfinx::fem::interpolation_matrix<T, U>(
                  V0, V1, A.mat_add_values(bs0, bs1));
            });

        return A;
      },
      nb::arg("V0"), nb::arg("V1"));

  m.def(
      "discrete_curl",
      [](const dolfinx::fem::FunctionSpace<U>& V0,
         const dolfinx::fem::FunctionSpace<U>& V1)
      {
        dolfinx::la::SparsityPattern sp
            = create_sparsity_entity_closure(V0, V1);

        // Build operator
        dolfinx::la::MatrixCSR<T> A(sp);
        dolfinx::fem::discrete_curl<U, T>(V0, V1, A.mat_set_values());
        return A;
      },
      nb::arg("V0"), nb::arg("V1"));

  m.def(
      "discrete_gradient",
      [](const dolfinx::fem::FunctionSpace<U>& V0,
         const dolfinx::fem::FunctionSpace<U>& V1)
      {
        dolfinx::la::SparsityPattern sp
            = create_sparsity_entity_closure(V0, V1);

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
void declare_assembly_functions(nanobind::module_& m)
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
        std::ranges::transform(
            coeffs, std::inserter(c, c.end()),
            [](auto& e) -> typename decltype(c)::value_type
            {
              std::size_t num_ents
                  = e.second.first.empty()
                        ? 0
                        : e.second.first.size() / e.second.second;
              return std::pair<std::pair<dolfinx::fem::IntegralType, int>,
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
      "pack_coefficients",
      [](const dolfinx::fem::Expression<T, U>& e,
         const dolfinx::mesh::Mesh<U>& mesh,
         nb::ndarray<const std::int32_t, nb::c_contig> entities)
      {
        auto [coeffs, cstride] = visit_entities(
            entities, [&e, &mesh](auto entities)
            { return dolfinx::fem::pack_coefficients(e, mesh, entities); });
        return dolfinx_wrappers::as_nbarray(
            std::move(coeffs),
            {entities.shape(0), static_cast<std::size_t>(cstride)});
      },
      nb::arg("expression"), nb::arg("mesh"), nb::arg("entities"),
      "Pack coefficients for a Expression.");
  m.def(
      "pack_constants",
      [](const dolfinx::fem::Form<T, U>& form)
      {
        return dolfinx_wrappers::as_nbarray(dolfinx::fem::pack_constants(form));
      },
      nb::arg("form"), "Pack constants for a Form.");
  m.def(
      "pack_constants", [](const dolfinx::fem::Expression<T, U>& e)
      { return dolfinx_wrappers::as_nbarray(dolfinx::fem::pack_constants(e)); },
      nb::arg("e"), "Pack constants for an Expression.");

  // Expression
  m.def(
      "tabulate_expression",
      [](nb::ndarray<T, nb::c_contig> values,
         const dolfinx::fem::Expression<T, U>& e,
         nb::ndarray<const T, nb::ndim<1>, nb::c_contig> constants,
         nb::ndarray<const T, nb::ndim<2>, nb::c_contig> coeffs,
         const dolfinx::mesh::Mesh<U>& mesh,
         nb::ndarray<const std::int32_t, nb::c_contig> entities)
      {
        std::optional<std::pair<
            std::reference_wrapper<const dolfinx::fem::FiniteElement<U>>,
            std::size_t>>
            element = std::nullopt;
        if (auto V = e.argument_space(); V)
        {
          std::size_t num_argument_dofs
              = V->dofmap()->element_dof_layout().num_dofs()
                * V->dofmap()->bs();
          assert(V->element());
          element = {std::cref(*V->element()), num_argument_dofs};
        }

        visit_entities(
            entities,
            [&values, &e, &coeffs, &constants, &mesh, &element](auto entities)
            {
              dolfinx::fem::tabulate_expression(
                  std::span<T>(values.data(), values.size()), e,
                  md::mdspan(coeffs.data(), coeffs.shape(0), coeffs.shape(1)),
                  std::span(constants.data(), constants.size()), mesh, entities,
                  element);
            });
      },
      nb::arg("values"), nb::arg("expression"), nb::arg("constants"),
      nb::arg("coefficients"), nb::arg("mesh"), nb::arg("entities"),
      "Evaluate an Expression on mesh entities.");
  m.def(
      "assemble_scalar",
      [](const dolfinx::fem::Form<T, U>& M,
         nb::ndarray<const T, nb::ndim<1>, nb::c_contig> constants,
         const std::map<std::pair<dolfinx::fem::IntegralType, int>,
                        nb::ndarray<const T, nb::ndim<2>, nb::c_contig>>&
             coefficients)
      {
        return dolfinx::fem::assemble_scalar(
            M, std::span(constants.data(), constants.size()),
            dolfinx_wrappers::py_to_cpp_coeffs(coefficients));
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
        dolfinx::fem::assemble_vector(
            std::span(b.data(), b.size()), L,
            std::span(constants.data(), constants.size()),
            dolfinx_wrappers::py_to_cpp_coeffs(coefficients));
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
         nb::ndarray<const std::int8_t, nb::ndim<1>, nb::c_contig> dof_marker0,
         nb::ndarray<const std::int8_t, nb::ndim<1>, nb::c_contig> dof_marker1)
      {
        std::span<const T> _constants(constants.data(), constants.size());
        const std::map<std::pair<dolfinx::fem::IntegralType, int>,
                       std::pair<std::span<const T>, int>>
            _coefficients = dolfinx_wrappers::py_to_cpp_coeffs(coefficients);
        std::span<const std::int8_t> _dof_marker0(dof_marker0.data(),
                                                  dof_marker0.size());
        std::span<const std::int8_t> _dof_marker1(dof_marker1.data(),
                                                  dof_marker1.size());

        // Element data block sizes are the DOF map block sizes, which
        // are the same for all DOF maps of a mixed-topology mesh
        visit_block_size({a.function_spaces().at(0)->dofmaps().front()->bs(),
                          a.function_spaces().at(1)->dofmaps().front()->bs()},
                         [&A, &a, _constants, &_coefficients, _dof_marker0,
                          _dof_marker1](auto bs0, auto bs1)
                         {
                           dolfinx::fem::assemble_matrix(
                               A.mat_add_values(bs0, bs1), a, _constants,
                               _coefficients, _dof_marker0, _dof_marker1);
                         });
      },
      nb::arg("A"), nb::arg("a"), nb::arg("constants"), nb::arg("coeffs"),
      nb::arg("dof_marker0"), nb::arg("dof_marker1"), "Experimental.");
  m.def(
      "assemble_matrix",
      [](std::function<int(
             nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig,
                         nb::numpy>,
             nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig,
                         nb::numpy>,
             nb::ndarray<const T, nb::ndim<2>, nb::c_contig, nb::numpy>)>
             fin,
         const dolfinx::fem::Form<T, U>& a,
         nb::ndarray<const T, nb::ndim<1>, nb::c_contig> constants,
         const std::map<std::pair<dolfinx::fem::IntegralType, int>,
                        nb::ndarray<const T, nb::ndim<2>, nb::c_contig>>&
             coefficients,
         nb::ndarray<const std::int8_t, nb::ndim<1>, nb::c_contig> dof_marker0,
         nb::ndarray<const std::int8_t, nb::ndim<1>, nb::c_contig> dof_marker1)
      {
        auto f = [&fin](const dolfinx::common::LocalIndexRange auto& rows,
                        const dolfinx::common::LocalIndexRange auto& cols,
                        std::span<const T> data)
        {
          const std::size_t num_rows = std::ranges::size(rows);
          const std::size_t num_cols = std::ranges::size(cols);
          return fin(
              nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig,
                          nb::numpy>(std::ranges::data(rows), {num_rows}),
              nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig,
                          nb::numpy>(std::ranges::data(cols), {num_cols}),
              nb::ndarray<const T, nb::ndim<2>, nb::c_contig, nb::numpy>(
                  data.data(), {num_rows, num_cols}));
        };
        dolfinx::fem::assemble_matrix(
            f, a, std::span(constants.data(), constants.size()),
            dolfinx_wrappers::py_to_cpp_coeffs(coefficients),
            std::span(dof_marker0.data(), dof_marker0.size()),
            std::span(dof_marker1.data(), dof_marker1.size()));
      },
      nb::arg("fin"), nb::arg("a"), nb::arg("constants"), nb::arg("coeffs"),
      nb::arg("dof_marker0"), nb::arg("dof_marker1"),
      "Experimental assembly with Python insertion function. This will be "
      "slow. Use for testing only.");

  // BC modifiers
  m.def(
      "apply_lifting",
      [](nb::ndarray<T, nb::ndim<1>, nb::c_contig> b,
         const std::vector<const dolfinx::fem::Form<T, U>*>& a,
         const std::vector<nb::ndarray<const T, nb::ndim<1>, nb::c_contig>>&
             constants,
         const std::vector<
             std::map<std::pair<dolfinx::fem::IntegralType, int>,
                      nb::ndarray<const T, nb::ndim<2>, nb::c_contig>>>& coeffs,
         const std::vector<nb::ndarray<const std::int8_t, nb::ndim<1>,
                                       nb::c_contig>>& bc_markers1,
         const std::vector<nb::ndarray<const T, nb::ndim<1>, nb::c_contig>>&
             bc_values1,
         const std::vector<nb::ndarray<const T, nb::ndim<1>, nb::c_contig>>& x0,
         T alpha)
      {
        std::vector<std::optional<
            std::reference_wrapper<const dolfinx::fem::Form<T, U>>>>
            _a;
        for (auto form : a)
        {
          if (form)
            _a.push_back(*form);
          else
            _a.push_back(std::nullopt);
        }

        std::vector<std::span<const T>> _x0 = vec_of_spans(x0);
        std::vector<std::span<const T>> _constants = vec_of_spans(constants);

        std::vector<std::map<std::pair<dolfinx::fem::IntegralType, int>,
                             std::pair<std::span<const T>, int>>>
            _coeffs;
        std::ranges::transform(
            coeffs, std::back_inserter(_coeffs),
            [](auto& c) { return dolfinx_wrappers::py_to_cpp_coeffs(c); });

        dolfinx::fem::apply_lifting(
            std::span<T>(b.data(), b.size()), _a, _constants, _coeffs,
            vec_of_spans(bc_markers1), vec_of_spans(bc_values1), _x0, alpha);
      },
      nb::arg("b").noconvert(), nb::arg("a"), nb::arg("constants"),
      nb::arg("coeffs"), nb::arg("bc_markers1"), nb::arg("bc_values1"),
      nb::arg("x0"), nb::arg("alpha"),
      "Modify vector for lifted boundary conditions");
}

} // namespace dolfinx_wrappers
