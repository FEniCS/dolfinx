// Copyright (C) 2021 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cstdint>
#include <dolfinx/io/cells.h>
#include <dolfinx/mesh/cell_types.h>
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

TEST_CASE("Prism and pyramid cell degree", "[io][cells]")
{
  using dolfinx::mesh::CellType;
  namespace cells = dolfinx::io::cells;
  CHECK(cells::cell_degree(CellType::prism, 6) == 1);
  CHECK(cells::cell_degree(CellType::prism, 18) == 2);
  CHECK(cells::cell_degree(CellType::pyramid, 5) == 1);
  CHECK(cells::cell_degree(CellType::pyramid, 14) == 2);
  CHECK_THROWS(cells::cell_degree(CellType::prism, 17));
  CHECK_THROWS(cells::cell_degree(CellType::pyramid, 15));

  // Serendipity, not Lagrange: a degree would collide with the 18-node
  // wedge and the 14-node pyramid, which VTKHDF reads keyed on degree
  CHECK_THROWS(cells::cell_degree(CellType::prism, 15));
  CHECK_THROWS(cells::cell_degree(CellType::pyramid, 13));
}

TEST_CASE("VTK cell type round-trip", "[io][cells]")
{
  using dolfinx::mesh::CellType;
  namespace cells = dolfinx::io::cells;

  // One layout per shape, with the identifier transcribed from
  // https://vtk.org/doc/nightly/html/vtkCellType_8h_source.html so that
  // the test is anchored to VTK and not only to the inverse map, which
  // could agree with a mistake.
  auto [cell, num_nodes, expected]
      = GENERATE(Catch::Generators::table<CellType, int, std::int8_t>(
          {{CellType::point, 1, 1},
           {CellType::interval, 3, 68},
           {CellType::triangle, 6, 69},
           {CellType::quadrilateral, 9, 70},
           {CellType::tetrahedron, 10, 71},
           {CellType::hexahedron, 27, 72},
           {CellType::prism, 18, 73},
           {CellType::pyramid, 5, 14}}));

  const std::int8_t vtk = cells::get_vtk_cell_type(cell, num_nodes);
  CHECK(vtk == expected);

  // The inverse must give back the shape and the layout it was handed.
  // VTK_PYRAMID carries its degree; the arbitrary-degree Lagrange types
  // carry none, reported as -1.
  auto [cell_out, degree] = cells::vtk_to_dolfinx(vtk);
  CHECK(cell_out == cell);
  if (cell == CellType::pyramid)
    CHECK(degree == cells::cell_degree(cell, num_nodes));
  else
    CHECK(degree == -1);
}

TEST_CASE("VTK pyramid layouts above linear", "[io][cells]")
{
  using dolfinx::mesh::CellType;
  namespace cells = dolfinx::io::cells;

  // VTK has no arbitrary-degree Lagrange pyramid, and its quadratic one
  // is the 13-node serendipity cell, which basix cannot express. So the
  // linear pyramid is the only one DOLFINx can write, and the rest are
  // rejected rather than mislabelled as it.
  CHECK_THROWS_AS(cells::get_vtk_cell_type(CellType::pyramid, 13),
                  std::invalid_argument);
  CHECK_THROWS_AS(cells::get_vtk_cell_type(CellType::pyramid, 14),
                  std::invalid_argument);
  CHECK_THROWS_AS(cells::get_vtk_cell_type(CellType::pyramid, 30),
                  std::invalid_argument);
}
