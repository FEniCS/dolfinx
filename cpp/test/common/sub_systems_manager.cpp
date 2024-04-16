// Copyright (C) 2015 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#ifdef HAS_PETSC
#include <petscvec.h>

namespace
{
void init_petsc()
{
  // Test user initialisation of PETSc
  int argc = 0;
  char** argv = nullptr;
  PetscInitialize(&argc, &argv, nullptr, nullptr);

  Vec x;
  VecCreate(MPI_COMM_WORLD, &x);
  VecDestroy(&x);
}
} // namespace

TEST_CASE("Initialise PETSc", "[petsc_init]") { CHECK_NOTHROW(init_petsc()); }
#endif
