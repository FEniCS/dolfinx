// Copyright (C) 2026 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "gpu_compiler.h"
#include <catch2/catch_test_macros.hpp>
#include <mpi.h>

TEST_CASE("DOLFINx code compiled by a GPU compiler", "[gpu]")
{
  dolfinx_test::GpuCompilerResults r
      = dolfinx_test::gpu_compiler_host_code(MPI_COMM_WORLD);
  CHECK(r.interpolation);
  CHECK(r.submesh);
  CHECK(r.assembly);
  CHECK(r.assembly_with_bc);
}
