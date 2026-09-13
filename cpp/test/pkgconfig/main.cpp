// Copyright (C) 2025 Jack S. Hale
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <dolfinx/common/log.h>
#include <dolfinx/mesh/generation.h>
#include <mpi.h>

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);
  dolfinx::init_logging(argc, argv);

  {
    auto mesh = dolfinx::mesh::create_rectangle(
        MPI_COMM_WORLD, {{{0.0, 0.0}, {1.0, 1.0}}}, {8, 8},
        dolfinx::mesh::CellType::triangle);
  }

  MPI_Finalize();
  return 0;
}
