// Copyright (C) 2021 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <dolfinx/io/cells.h>
#include <stdexcept>

#ifdef HAS_ADIOS2

#include <algorithm>
#include <concepts>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/fem/functionspace_factory.h>
#include <dolfinx/io/ADIOS2Writers.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/generation.h>
#include <format>
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
      mesh,
      std::make_shared<fem::FiniteElement<T>>(e, mesh->geometry().dim())));

  // Create a finite element Function
  auto u = std::make_shared<fem::Function<T>>(V);
  auto v = std::make_shared<fem::Function<std::complex<T>>>(V);

  std::filesystem::path f = std::format("test_vtx_reuse_mesh{}.bp", sizeof(T));
  io::VTXWriter<T> writer(mesh->comm(), f, {u, v}, "BPFile",
                          io::VTXMeshPolicy::reuse);
  writer.write(0);

  std::ranges::fill(u->x()->array(), 1);

  writer.write(1);
}
} // namespace

TEST_CASE("VTX reuse mesh")
{
  CHECK_NOTHROW(test_vtx_reuse_mesh<float>());
  CHECK_NOTHROW(test_vtx_reuse_mesh<double>());
}

#endif

TEST_CASE("Prism and pyramid IO layouts", "[io][cells]")
{
  using dolfinx::mesh::CellType;
  namespace cells = dolfinx::io::cells;
  CHECK(cells::cell_degree(CellType::prism, 6) == 1);
  CHECK(cells::cell_degree(CellType::prism, 15) == 2);
  CHECK(cells::cell_degree(CellType::prism, 18) == 2);
  CHECK(cells::cell_degree(CellType::pyramid, 5) == 1);
  CHECK(cells::cell_degree(CellType::pyramid, 13) == 2);
  CHECK(cells::cell_degree(CellType::pyramid, 14) == 2);
  CHECK_THROWS(cells::cell_degree(CellType::prism, 17));
  CHECK_THROWS(cells::cell_degree(CellType::pyramid, 15));

  CHECK(cells::get_vtk_cell_type(CellType::pyramid, 3, 5) == 14);
  CHECK(cells::get_vtk_cell_type(CellType::pyramid, 3, 13) == 27);
  CHECK(cells::get_vtk_cell_type(CellType::prism, 3, 18) == 73);
  CHECK_THROWS_AS(cells::get_vtk_cell_type(CellType::pyramid, 3, 14),
                  std::invalid_argument);
  CHECK_THROWS_AS(cells::get_vtk_cell_type(CellType::pyramid, 2),
                  std::invalid_argument);
  CHECK_THROWS_AS(cells::get_vtk_cell_type(CellType::prism, 2),
                  std::invalid_argument);
  CHECK(cells::get_vtk_cell_type(CellType::triangle, 2) == 69);
  CHECK(cells::get_vtk_cell_type(CellType::quadrilateral, 2) == 70);
}
