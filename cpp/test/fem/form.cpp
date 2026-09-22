// Copyright (C) 2025-2026 Garth N. Wells and Jørgen S. Dokken
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "expr.h"
#include <basix/finite-element.h>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <dolfinx/fem/Expression.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/fem/assembler.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/generation.h>
#include <dolfinx/mesh/utils.h>

using namespace dolfinx;

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
          mesh,
          std::make_shared<fem::FiniteElement<double>>(
              element, mesh->geometry().dim(), std::vector<std::size_t>{3})));

  test_form_cmap_compat(V);
  test_expression_cmap_compat(V);
}

TEST_CASE("Form with data on a mesh of the wrong dimension",
          "[form_entity_domain]")
{
  // `Form` maps an integration entity to a *cell* of the argument's
  // mesh, so a ridge integral cannot take data on a facet submesh. FFCx
  // rejects the combination when compiling a kernel, so this is reached
  // only by building a Form directly, as here.
  auto mesh = std::make_shared<mesh::Mesh<double>>(mesh::create_box<double>(
      MPI_COMM_WORLD, {{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}}, {2, 2, 2},
      mesh::CellType::tetrahedron, graph::partition_graph));
  const int tdim = mesh->topology()->dim();
  mesh->topology_mutable()->create_entities(tdim - 1);
  mesh->topology_mutable()->create_entities(tdim - 2);
  mesh->topology_mutable()->create_connectivity(tdim, tdim - 2);

  // A submesh of the facets, i.e. one dimension too high for a ridge
  auto facets = mesh::locate_entities(
      *mesh, tdim - 1,
      [](auto x) { return std::vector<std::int8_t>(x.extent(1), 1); });
  auto [submesh, e_map, v_map, g_map]
      = mesh::create_submesh(*mesh, tdim - 1, facets);
  auto smesh = std::make_shared<mesh::Mesh<double>>(std::move(submesh));

  auto element = basix::create_element<double>(
      basix::element::family::P, basix::cell::type::triangle, 1,
      basix::element::lagrange_variant::unset,
      basix::element::dpc_variant::unset, false);
  auto V = std::make_shared<fem::FunctionSpace<double>>(
      fem::create_functionspace<double>(
          smesh, std::make_shared<fem::FiniteElement<double>>(
                     element, std::vector<std::size_t>{})));

  auto kernel = [](double*, const double*, const double*, const double*,
                   const int*, const uint8_t*, void*) {};
  std::map integrals{
      std::pair{std::tuple{fem::IntegralType::ridge, -1, 0},
                fem::integral_data<double>(kernel, std::vector<std::int32_t>{},
                                           std::vector<int>{})}};

  auto build_form = [&V, &integrals, &mesh, &e_map]()
  {
    return fem::Form<double, double>({V}, integrals, mesh, {}, {}, false,
                                     {std::cref(e_map)});
  };

  CHECK_THROWS_WITH(build_form(), Catch::Matchers::ContainsSubstring(
                                      "integration entities of dimension"));
}
