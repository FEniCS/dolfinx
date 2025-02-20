// Copyright (C) 2025 Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "expr.h"
#include <basix/finite-element.h>
#include <catch2/catch_test_macros.hpp>
#include <dolfinx/fem/Expression.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/fem/assembler.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/generation.h>

using namespace dolfinx;
namespace md = MDSPAN_IMPL_STANDARD_NAMESPACE;

namespace
{
void test_form_cmap_compat(auto V)
{
  fem::create_form<double>(*form_expr_L1, {V}, {}, {}, {}, {});
  CHECK_THROWS(fem::create_form<double>(*form_expr_L2, {V}, {}, {}, {}, {}));
}

void test_expression_cmap_compat(auto V)
{
  auto u = std::make_shared<fem::Function<double>>(V);
  auto mesh = u->function_space()->mesh();

  std::vector<std::int32_t> cells(1);

  // Create Expression that expects P1 geometry
  dolfinx::fem::Expression<double> expr1
      = dolfinx::fem::create_expression<double>(*expression_expr_Q6_P1,
                                                {{"u1", u}}, {});
  auto [Xc, Xshape] = expr1.X();
  std::vector<double> grad_e(3 * Xshape[0] * cells.size());
  fem::tabulate_expression(std::span(grad_e), expr1, *mesh,
                           md::mdspan(cells.data(), cells.size()));

  // Create Expression that expects P2 geometry. Should throw because
  // mesh is P1.
  CHECK_THROWS(dolfinx::fem::create_expression<double>(*expression_expr_Q6_P2,
                                                       {{"u2", u}}, {}));
}
} // namespace

TEST_CASE("Create Expression/Form (mismatch of mesh geometry)",
          "[geometry_compat]")
{
  auto mesh = std::make_shared<mesh::Mesh<double>>(mesh::create_box<double>(
      MPI_COMM_WORLD, {{{0.0, 0.0, 0.0}, {0.96, 4.5, 2.0}}}, {2, 4, 5},
      mesh::CellType::hexahedron,
      mesh::create_cell_partitioner(mesh::GhostMode::none)));
  auto element = basix::create_element<double>(
      basix::element::family::P, basix::cell::type::hexahedron, 1,
      basix::element::lagrange_variant::unset,
      basix::element::dpc_variant::unset, false);

  auto V = std::make_shared<fem::FunctionSpace<double>>(
      fem::create_functionspace<double>(
          mesh, std::make_shared<fem::FiniteElement<double>>(
                    element, std::vector<std::size_t>{3})));

  test_form_cmap_compat(V);
  test_expression_cmap_compat(V);
}
