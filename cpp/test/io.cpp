// Copyright (C) 2021 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#ifdef HAS_ADIOS2

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <concepts>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/io/ADIOS2Writers.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/generation.h>
#include <mpi.h>

using namespace dolfinx;

namespace
{
template <std::floating_point T>
void test_vtx_reuse_mesh()
{
  auto mesh = std::make_shared<mesh::Mesh<T>>(
      mesh::create_rectangle<T>(MPI_COMM_WORLD, {{{0.0, 0.0}, {1.0, 1.0}}},
                                {22, 12}, mesh::CellType::triangle));

  // Create a Basix continuous Lagrange element of degree 1
  basix::FiniteElement e = basix::create_element<T>(
      basix::element::family::P,
      mesh::cell_type_to_basix_type(mesh::CellType::triangle), 1,
      basix::element::lagrange_variant::unset,
      basix::element::dpc_variant::unset, false);

  // Create a scalar function space
  auto V = std::make_shared<fem::FunctionSpace<T>>(fem::create_functionspace<T>(
      mesh, std::make_shared<fem::FiniteElement<T>>(e)));

  // Create a finite element Function
  auto u = std::make_shared<fem::Function<T>>(V);
  auto v = std::make_shared<fem::Function<std::complex<T>>>(V);

  std::filesystem::path f
      = "test_vtx_reuse_mesh" + std::to_string(sizeof(T)) + ".bp";
  io::VTXWriter<T> writer(mesh->comm(), f, {u, v}, "BPFile",
                          io::VTXMeshPolicy::reuse);
  writer.write(0);

  std::ranges::fill(u->x()->mutable_array(), 1);

  writer.write(1);
}
} // namespace

TEST_CASE("VTX reuse mesh")
{
  CHECK_NOTHROW(test_vtx_reuse_mesh<float>());
  CHECK_NOTHROW(test_vtx_reuse_mesh<double>());
}

#endif
