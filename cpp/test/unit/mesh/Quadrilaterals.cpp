// Copyright (C) 2019 Jorgen S. Dokken
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later
//
// Unit tests for Quadrilateral meshes

#include "quadrilateral.h"
#include <catch.hpp>
#include <cmath>
#include <dolfin.h>
#include <dolfin/io/XDMFFile.h>
#include <dolfin/io/cells.h>
#include <ufc.h>

using namespace dolfin;

namespace
{
void test_quadrilateral_mesh()
{

  const double L = 1;
  const double H = 1;
  const double Z = 0;

  const std::vector<std::int64_t> gci;
  std::int32_t ngp = 0;

  Eigen::Array<double, 9, 3, Eigen::RowMajor> points;
  points.row(0) << 0, 0, 0;
  points.row(1) << L, 0, 0;
  points.row(2) << L, H, Z;
  points.row(3) << 0, H, Z;
  points.row(4) << L / 2, 0, 0;
  points.row(5) << L, H / 2, 0;
  points.row(6) << L / 2, H, Z;
  points.row(7) << 0, H / 2, 0;
  points.row(8) << L / 2, H / 2, 0;

  Eigen::Array<std::int64_t, 1, 9, Eigen::RowMajor> cells;
  cells.row(0) << 0, 1, 2, 3, 4, 5, 6, 7, 8;
  Eigen::Array<std::int64_t, 1, 9, Eigen::RowMajor> cells_dolfin
      = io::cells::vtk_to_dolfin_ordering(cells, mesh::CellType::quadrilateral);
  auto mesh = std::make_shared<mesh::Mesh>(
      mesh::Mesh(MPI_COMM_WORLD, mesh::CellType::quadrilateral, points,
                 cells_dolfin, gci, mesh::GhostMode::none, ngp));
  io::XDMFFile outfile(MPI_COMM_WORLD, "mesh.xdmf");
  outfile.write(*mesh);
  ufc_form* form_local;

  ufc_coordinate_mapping* cmap = form_local->create_coordinate_mapping();
  auto dmap = dolfin::fem::get_cmap_from_ufc_cmap(*cmap);
  mesh->geometry().coord_mapping = dmap;

  auto V = fem::create_functionspace(quadrilateral_coefficientspace_u_create,
                                     mesh);
  auto f = std::make_shared<function::Function>(V);
  f->interpolate(
      [](auto x) { return (Eigen::sin(x.col(1)) + Eigen::sin(x.col(0))); });

  ufc_form* integral = quadrilateral_form_a_create();
  auto a = std::make_shared<fem::Form>(fem::create_form(*integral, {}));

  a->set_coefficients({{"f", f}});
  auto du = dolfin::fem::assemble_scalar(*a);
  int argc = 0;

  CHECK(argc == 1);
} // namespace
} // namespace

TEST_CASE("Quadrilaterals", "[quadrilateral_mesh]")
{
  CHECK_NOTHROW(test_quadrilateral_mesh());
}
