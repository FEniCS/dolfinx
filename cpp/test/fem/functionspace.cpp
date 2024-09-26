// Copyright (C) 2024 Paul T. Kühner
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include <basix/finite-element.h>

#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/generation.h>

using namespace dolfinx;

TEST_CASE("Create Function Space (mismatch of elements)", "[functionspace]")
{
  auto mesh = std::make_shared<mesh::Mesh<double>>(
      dolfinx::mesh::create_rectangle<double>(
          MPI_COMM_SELF, {{{0, 0}, {1, 1}}}, {1, 1}, mesh::CellType::triangle));

  auto element = basix::create_element<double>(
      basix::element::family::P, basix::cell::type::interval, 1,
      basix::element::lagrange_variant::unset,
      basix::element::dpc_variant::unset, false);

  CHECK_THROWS(fem::create_functionspace<double>(mesh, element, {}));
}
