// Copyright (C) 2026 Jack S. Hale
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later
//
// Unit tests for the la::petsc wrappers

#ifdef HAS_PETSC

#include <catch2/catch_test_macros.hpp>
#include <dolfinx/la/petsc.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petscvec.h>
#include <stdexcept>
#include <utility>

using namespace dolfinx;

namespace
{
/// PetscInitialize returns immediately if PETSc is already initialised
void init_petsc()
{
  int argc = 0;
  char** argv = nullptr;
  REQUIRE(PetscInitialize(&argc, &argv, nullptr, nullptr) == 0);
}
} // namespace

TEST_CASE("PETSc wrappers reject null handles", "[petsc]")
{
  init_petsc();

  // Wrapping a null handle stored it and then dereferenced it in
  // PetscObjectReference, where PetscValidHeaderSpecific is a no-op in
  // optimised PETSc builds
  CHECK_THROWS_AS(la::petsc::Vector(nullptr, true), std::runtime_error);
  CHECK_THROWS_AS(la::petsc::Operator(nullptr, true), std::runtime_error);
  CHECK_THROWS_AS(la::petsc::KrylovSolver(nullptr, true), std::runtime_error);

  // inc_ref_count = false takes the same path
  CHECK_THROWS_AS(la::petsc::KrylovSolver(nullptr, false), std::runtime_error);
}

TEST_CASE("PETSc Krylov solver", "[petsc]")
{
  init_petsc();

  constexpr PetscInt N = 10;

  // Identity matrix, so the solution of Ax = b is b
  Mat A = nullptr;
  MatCreate(MPI_COMM_WORLD, &A);
  MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, N, N);
  MatSetType(A, MATAIJ);
  MatSetUp(A);
  PetscInt r0 = 0, r1 = 0;
  MatGetOwnershipRange(A, &r0, &r1);
  for (PetscInt i = r0; i < r1; ++i)
  {
    PetscScalar v = 1.0;
    MatSetValues(A, 1, &i, 1, &i, &v, INSERT_VALUES);
  }
  MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

  SECTION("Solve returns the iteration count")
  {
    la::petsc::KrylovSolver solver(MPI_COMM_WORLD);
    solver.set_operator(A);

    Vec x = nullptr, b = nullptr;
    MatCreateVecs(A, &x, &b);
    VecSet(b, 1.0);

    int num_it = solver.solve(x, b, false);
    CHECK(num_it >= 0);

    // A is the identity, so x == b
    PetscReal norm = 0;
    VecAXPY(x, -1.0, b);
    VecNorm(x, NORM_2, &norm);
    CHECK(static_cast<double>(norm) < 1e-10);

    VecDestroy(&x);
    VecDestroy(&b);
  }

  SECTION("Solving before setting an operator throws")
  {
    // KSPGetOperators cannot detect this: PCGetOperators creates an
    // empty Mat when none is set, so it never reports null. The error
    // comes from KSPSolve failing on that untyped Mat
    la::petsc::KrylovSolver solver(MPI_COMM_WORLD);
    Vec x = nullptr, b = nullptr;
    MatCreateVecs(A, &x, &b);
    VecSet(b, 1.0);
    CHECK_THROWS_AS(solver.solve(x, b, false), std::runtime_error);
    VecDestroy(&x);
    VecDestroy(&b);
  }

  SECTION("Options prefix round-trips")
  {
    la::petsc::KrylovSolver solver(MPI_COMM_WORLD);
    solver.set_options_prefix("mysolver_");
    CHECK(solver.get_options_prefix() == "mysolver_");
  }

  SECTION("Move leaves the source safe to destroy")
  {
    la::petsc::KrylovSolver a(MPI_COMM_WORLD);
    KSP raw = a.ksp();
    la::petsc::KrylovSolver b(std::move(a));
    CHECK(b.ksp() == raw);
    CHECK(a.ksp() == nullptr);
  }

  MatDestroy(&A);
}

#endif
