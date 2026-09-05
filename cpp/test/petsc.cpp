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
  // optimised PETSc builds. The null check happens before inc_ref_count
  // is examined, so both values must be rejected.
  CHECK_THROWS_AS(la::petsc::Vector(nullptr, true), std::runtime_error);
  CHECK_THROWS_AS(la::petsc::Vector(nullptr, false), std::runtime_error);
  CHECK_THROWS_AS(la::petsc::Matrix(nullptr, true), std::runtime_error);
  CHECK_THROWS_AS(la::petsc::Matrix(nullptr, false), std::runtime_error);
  CHECK_THROWS_AS(la::petsc::KrylovSolver(nullptr, true), std::runtime_error);
  CHECK_THROWS_AS(la::petsc::KrylovSolver(nullptr, false), std::runtime_error);
}

TEST_CASE("PETSc wrappers manage reference counts", "[petsc]")
{
  init_petsc();

  PetscInt refs = 0;

  SECTION("Matrix, inc_ref_count = true shares ownership")
  {
    Mat A = nullptr;
    CHECK(MatCreate(MPI_COMM_WORLD, &A) == 0);
    CHECK(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, 1, 1) == 0);
    CHECK(MatSetType(A, MATAIJ) == 0);
    CHECK(MatSetUp(A) == 0);
    CHECK(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY) == 0);
    CHECK(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY) == 0);

    {
      la::petsc::Matrix op(A, true);
      CHECK(PetscObjectGetReference((PetscObject)A, &refs) == 0);
      CHECK(refs == 2);
    }

    // Wrapper released its share; A must still be alive and usable
    CHECK(PetscObjectGetReference((PetscObject)A, &refs) == 0);
    CHECK(refs == 1);
    CHECK(MatDestroy(&A) == 0);
  }

  SECTION("Matrix, inc_ref_count = false takes ownership")
  {
    Mat A = nullptr;
    CHECK(MatCreate(MPI_COMM_WORLD, &A) == 0);
    CHECK(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, 1, 1) == 0);
    CHECK(MatSetType(A, MATAIJ) == 0);
    CHECK(MatSetUp(A) == 0);
    CHECK(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY) == 0);
    CHECK(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY) == 0);

    // No caller-side reference remains once the wrapper is constructed;
    // destroying it must be the only thing that frees A (nothing left
    // here to double-destroy).
    la::petsc::Matrix op(A, false);
    CHECK(PetscObjectGetReference((PetscObject)op.mat(), &refs) == 0);
    CHECK(refs == 1);
  }

  SECTION("Vector, inc_ref_count = true shares ownership")
  {
    Vec x = nullptr;
    CHECK(VecCreate(MPI_COMM_WORLD, &x) == 0);
    CHECK(VecSetSizes(x, PETSC_DECIDE, 1) == 0);
    CHECK(VecSetFromOptions(x) == 0);

    {
      la::petsc::Vector v(x, true);
      CHECK(PetscObjectGetReference((PetscObject)x, &refs) == 0);
      CHECK(refs == 2);
    }

    CHECK(PetscObjectGetReference((PetscObject)x, &refs) == 0);
    CHECK(refs == 1);
    CHECK(VecDestroy(&x) == 0);
  }

  SECTION("KrylovSolver, inc_ref_count = true shares ownership")
  {
    KSP ksp = nullptr;
    CHECK(KSPCreate(MPI_COMM_WORLD, &ksp) == 0);

    {
      la::petsc::KrylovSolver solver(ksp, true);
      CHECK(PetscObjectGetReference((PetscObject)ksp, &refs) == 0);
      CHECK(refs == 2);
    }

    CHECK(PetscObjectGetReference((PetscObject)ksp, &refs) == 0);
    CHECK(refs == 1);
    CHECK(KSPDestroy(&ksp) == 0);
  }
}

TEST_CASE("PETSc Krylov solver", "[petsc]")
{
  init_petsc();

  constexpr PetscInt N = 10;

  // Identity matrix, so the solution of Ax = b is b
  Mat A = nullptr;
  CHECK(MatCreate(MPI_COMM_WORLD, &A) == 0);
  CHECK(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, N, N) == 0);
  CHECK(MatSetType(A, MATAIJ) == 0);
  CHECK(MatSetUp(A) == 0);
  PetscInt r0 = 0, r1 = 0;
  CHECK(MatGetOwnershipRange(A, &r0, &r1) == 0);
  for (PetscInt i = r0; i < r1; ++i)
  {
    PetscScalar v = 1.0;
    CHECK(MatSetValues(A, 1, &i, 1, &i, &v, INSERT_VALUES) == 0);
  }
  CHECK(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY) == 0);
  CHECK(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY) == 0);

  SECTION("Solve returns the convergence reason")
  {
    la::petsc::KrylovSolver solver(MPI_COMM_WORLD);
    solver.set_operator(A);

    Vec x = nullptr, b = nullptr;
    CHECK(MatCreateVecs(A, &x, &b) == 0);
    CHECK(VecSet(b, 1.0) == 0);

    CHECK(solver.solve(x, b, false) > 0);

    // The number of iterations is available via the raw PETSc API
    PetscInt num_it = -1;
    CHECK(KSPGetIterationNumber(solver.ksp(), &num_it) == 0);
    CHECK(num_it >= 0);

    // A is the identity, so x == b
    PetscReal norm = 0;
    CHECK(VecAXPY(x, -1.0, b) == 0);
    CHECK(VecNorm(x, NORM_2, &norm) == 0);
    CHECK(norm < 100 * PETSC_SMALL);

    CHECK(VecDestroy(&x) == 0);
    CHECK(VecDestroy(&b) == 0);
  }

  SECTION("ksp() exposes a handle usable with the raw PETSc API")
  {
    // Bypass the class entirely and drive ksp() with the raw PETSc API
    la::petsc::KrylovSolver solver(MPI_COMM_WORLD);
    CHECK(KSPSetOperators(solver.ksp(), A, A) == 0);

    Vec x = nullptr, b = nullptr;
    CHECK(MatCreateVecs(A, &x, &b) == 0);
    CHECK(VecSet(b, 1.0) == 0);

    CHECK(KSPSolve(solver.ksp(), b, x) == 0);

    KSPConvergedReason reason;
    CHECK(KSPGetConvergedReason(solver.ksp(), &reason) == 0);
    CHECK(reason > 0);

    // A is the identity, so x == b
    PetscReal norm = 0;
    CHECK(VecAXPY(x, -1.0, b) == 0);
    CHECK(VecNorm(x, NORM_2, &norm) == 0);
    CHECK(norm < 100 * PETSC_SMALL);

    CHECK(VecDestroy(&x) == 0);
    CHECK(VecDestroy(&b) == 0);
  }

  SECTION("Solving the transpose system")
  {
    // A is symmetric (identity), so the transpose solve has the same
    // solution and exercises the KSPSolveTranspose path in solve()
    la::petsc::KrylovSolver solver(MPI_COMM_WORLD);
    solver.set_operator(A);

    Vec x = nullptr, b = nullptr;
    CHECK(MatCreateVecs(A, &x, &b) == 0);
    CHECK(VecSet(b, 1.0) == 0);

    CHECK(solver.solve(x, b, true) > 0);

    PetscReal norm = 0;
    CHECK(VecAXPY(x, -1.0, b) == 0);
    CHECK(VecNorm(x, NORM_2, &norm) == 0);
    CHECK(norm < 100 * PETSC_SMALL);

    CHECK(VecDestroy(&x) == 0);
    CHECK(VecDestroy(&b) == 0);
  }

  SECTION("Solving with a distinct preconditioner matrix")
  {
    la::petsc::KrylovSolver solver(MPI_COMM_WORLD);
    solver.set_operators(A, A);

    Vec x = nullptr, b = nullptr;
    CHECK(MatCreateVecs(A, &x, &b) == 0);
    CHECK(VecSet(b, 1.0) == 0);

    CHECK(solver.solve(x, b, false) > 0);

    CHECK(VecDestroy(&x) == 0);
    CHECK(VecDestroy(&b) == 0);
  }

  SECTION("Solving through a wrapped external KSP handle")
  {
    // KrylovSolver(ksp, inc_ref_count) must produce a wrapper that is
    // fully usable via solve(), not just safe to destroy
    for (bool inc_ref_count : {true, false})
    {
      KSP ksp = nullptr;
      CHECK(KSPCreate(MPI_COMM_WORLD, &ksp) == 0);
      CHECK(KSPSetOperators(ksp, A, A) == 0);

      la::petsc::KrylovSolver solver(ksp, inc_ref_count);

      Vec x = nullptr, b = nullptr;
      CHECK(MatCreateVecs(A, &x, &b) == 0);
      CHECK(VecSet(b, 1.0) == 0);

      CHECK(solver.solve(x, b, false) > 0);

      PetscReal norm = 0;
      CHECK(VecAXPY(x, -1.0, b) == 0);
      CHECK(VecNorm(x, NORM_2, &norm) == 0);
      CHECK(norm < 100 * PETSC_SMALL);

      CHECK(VecDestroy(&x) == 0);
      CHECK(VecDestroy(&b) == 0);

      if (inc_ref_count)
        CHECK(KSPDestroy(&ksp) == 0);
    }
  }

  SECTION("Solving before setting an operator throws")
  {
    // KSPGetOperators cannot detect this: PCGetOperators creates an
    // empty Mat when none is set, so it never reports null. The error
    // comes from KSPSolve failing on that untyped Mat
    la::petsc::KrylovSolver solver(MPI_COMM_WORLD);
    Vec x = nullptr, b = nullptr;
    CHECK(MatCreateVecs(A, &x, &b) == 0);
    CHECK(VecSet(b, 1.0) == 0);
    CHECK_THROWS_AS(solver.solve(x, b, false), std::runtime_error);
    CHECK(VecDestroy(&x) == 0);
    CHECK(VecDestroy(&b) == 0);
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

  SECTION("Move assignment releases the target's prior handle")
  {
    la::petsc::KrylovSolver a(MPI_COMM_WORLD);
    la::petsc::KrylovSolver b(MPI_COMM_WORLD);
    KSP raw_a = a.ksp();
    KSP raw_b = b.ksp();

    PetscInt refs = 0;
    CHECK(PetscObjectGetReference((PetscObject)raw_b, &refs) == 0);
    CHECK(refs == 1);

    b = std::move(a);
    CHECK(b.ksp() == raw_a);
    CHECK(a.ksp() == raw_b);

    // a now holds b's former handle and destroys it going out of scope;
    // nothing else references raw_b, so it must not leak
    CHECK(PetscObjectGetReference((PetscObject)raw_b, &refs) == 0);
    CHECK(refs == 1);
  }

  CHECK(MatDestroy(&A) == 0);
}

#endif
