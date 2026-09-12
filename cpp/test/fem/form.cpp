// Copyright (C) 2025-2026 Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "expr.h"
#include <basix/finite-element.h>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <dolfinx/fem/Expression.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/fem/assembler.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/generation.h>
#include <map>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

using namespace dolfinx;

using Form = fem::Form<double>;
static_assert(not std::is_copy_constructible_v<Form>);
static_assert(std::is_move_constructible_v<Form>);
static_assert(not std::is_copy_assignable_v<Form>);
static_assert(std::is_move_assignable_v<Form>);
static_assert(noexcept(std::declval<const Form&>().rank()));
static_assert(noexcept(std::declval<const Form&>().mesh()));
static_assert(noexcept(std::declval<const Form&>().function_spaces()));
static_assert(noexcept(std::declval<const Form&>().coefficients()));
static_assert(noexcept(std::declval<const Form&>().constants()));
static_assert(noexcept(std::declval<const Form&>().needs_facet_permutations()));

namespace
{
void test_form_cmap_compat(const auto& V)
{
  fem::create_form<double>(*form_expr_L1, {V}, {}, {}, {}, {});
  CHECK_THROWS(fem::create_form<double>(*form_expr_L2, {V}, {}, {}, {}, {}));
}

void test_expression_cmap_compat(const auto& V)
{
  auto u = std::make_shared<fem::Function<double>>(V);
  auto mesh = u->function_space()->mesh();

  std::vector<std::int32_t> cells(1);

  // Create Expression that expects P1 geometry
  dolfinx::fem::Expression<double> expr1
      = dolfinx::fem::create_expression<double>(*expression_expr_Q6_P1,
                                                {{"u1", u}}, {}, {});
  auto [Xc, Xshape] = expr1.X();
  std::vector<double> grad_e(3 * Xshape[0] * cells.size());
  fem::tabulate_expression(std::span(grad_e), expr1, *mesh,
                           md::mdspan(cells.data(), cells.size()));

  // Create Expression that expects P2 geometry. Should throw because
  // mesh is P1.
  dolfinx::fem::Expression<double> expr2
      = dolfinx::fem::create_expression<double>(*expression_expr_Q6_P2,
                                                {{"u2", u}}, {}, {});
  CHECK_THROWS(fem::tabulate_expression(
      std::span(grad_e), expr2, *mesh, md::mdspan(cells.data(), cells.size())));
}
} // namespace

TEST_CASE("Create Expression/Form (mismatch of mesh geometry)",
          "[geometry_compat]")
{
  auto mesh = std::make_shared<mesh::Mesh<double>>(mesh::create_box<double>(
      MPI_COMM_WORLD, {{{0.0, 0.0, 0.0}, {0.96, 4.5, 2.0}}}, {2, 4, 5},
      mesh::CellType::hexahedron, graph::partition_graph));
  auto element = basix::create_element<double>(
      basix::element::family::P, basix::cell::type::hexahedron, 1,
      basix::element::lagrange_variant::unset,
      basix::element::dpc_variant::unset, false);

  auto V = std::make_shared<fem::FunctionSpace<double>>(
      fem::create_functionspace<double>(
          mesh, std::make_shared<fem::FiniteElement<double>>(
                    element, std::vector<std::size_t>{3})));

  const std::map<std::tuple<fem::IntegralType, int, int>,
                 fem::integral_data<double>>
      integrals;
  const std::vector<std::shared_ptr<const fem::FunctionSpace<double>>>
      null_spaces(1);
  CHECK_THROWS_AS(Form(null_spaces, integrals, mesh, {}, {}, false, {}),
                  std::invalid_argument);

  const std::vector<std::shared_ptr<const fem::Function<double>>>
      null_coefficients(1);
  const std::map<std::tuple<fem::IntegralType, int, int>,
                 fem::integral_data<double>>
      coefficient_integrals{
          {{fem::IntegralType::cell, 0, 0},
           fem::integral_data<double>(
               [](double*, const double*, const double*, const double*,
                  const int*, const std::uint8_t*, void*) {},
               std::vector<std::int32_t>{}, std::vector<int>{0})}};
  CHECK_THROWS_AS(
      Form({V}, coefficient_integrals, mesh, null_coefficients, {}, false, {}),
      std::invalid_argument);

  test_form_cmap_compat(V);
  test_expression_cmap_compat(V);
}
