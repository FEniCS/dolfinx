// Copyright (C) 2026 Jack S. Hale
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later
//
// Unit tests for nls::petsc::NonlinearProblem

#include <catch2/catch_test_macros.hpp>

#ifdef HAS_PETSC

#include <cmath>
#include <dolfinx/nls/NonlinearProblem.h>
#include <mpi.h>
#include <petscmat.h>
#include <petscsnes.h>
#include <petscsys.h>
#include <petscvec.h>
#include <span>
#include <string>
#include <utility>

using namespace dolfinx;

namespace
{
constexpr PetscInt local_size = 2;

// Residual of the decoupled system f(x)_i = x_i^2 - 2, which has the
// positive root x_i = sqrt(2)
void assemble_residual(const Vec x, Vec b)
{
  const PetscScalar* _x;
  VecGetArrayRead(x, &_x);
  PetscScalar* _b;
  VecGetArray(b, &_b);
  std::span<const PetscScalar> xs(_x, local_size);
  std::span<PetscScalar> bs(_b, local_size);
  for (PetscInt i = 0; i < local_size; ++i)
    bs[i] = xs[i] * xs[i] - PetscScalar(2);
  VecRestoreArray(b, &_b);
  VecRestoreArrayRead(x, &_x);
}

// Jacobian df/dx = diag(2 x_i)
void assemble_jacobian(const Vec x, Mat J)
{
  PetscInt r0, r1;
  VecGetOwnershipRange(x, &r0, &r1);
  const PetscScalar* _x;
  VecGetArrayRead(x, &_x);
  std::span<const PetscScalar> xs(_x, local_size);
  MatZeroEntries(J);
  for (PetscInt i = 0; i < local_size; ++i)
  {
    PetscInt row = r0 + i;
    PetscScalar v = PetscScalar(2) * xs[i];
    MatSetValues(J, 1, &row, 1, &row, &v, INSERT_VALUES);
  }
  VecRestoreArrayRead(x, &_x);
  MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY);
}

// Check that all entries of x are the positive root sqrt(2)
void check_solution(const Vec x)
{
  const PetscScalar* _x;
  VecGetArrayRead(x, &_x);
  std::span<const PetscScalar> xs(_x, local_size);
  for (PetscScalar xi : xs)
    CHECK(std::abs(PetscRealPart(xi) - std::sqrt(2.0)) < 1e-6);
  VecRestoreArrayRead(x, &_x);
}
} // namespace

TEST_CASE("Solve nonlinear problem with SNES", "[nls_snes]")
{
  int argc = 0;
  char** argv = nullptr;
  PetscInitialize(&argc, &argv, nullptr, nullptr);

  MPI_Comm comm = MPI_COMM_WORLD;
  Vec x, b;
  VecCreateMPI(comm, local_size, PETSC_DETERMINE, &x);
  VecDuplicate(x, &b);
  VecSet(x, PetscScalar(1));

  Mat J;
  MatCreateAIJ(comm, local_size, local_size, PETSC_DETERMINE, PETSC_DETERMINE,
               1, nullptr, 0, nullptr, &J);

  SECTION("Solve")
  {
    nls::petsc::NonlinearProblem problem(comm);
    problem.set_F(assemble_residual, b);
    problem.set_J([](const Vec x, Mat Jmat, Mat)
                  { assemble_jacobian(x, Jmat); }, J);

    problem.set_options_prefix("test_snes_");
    CHECK(problem.get_options_prefix() == "test_snes_");
    problem.set_from_options();

    int num_it = problem.solve(x);
    CHECK(num_it > 0);
    SNESConvergedReason reason;
    SNESGetConvergedReason(problem.snes(), &reason);
    CHECK(reason > 0);
    check_solution(x);
  }

  SECTION("Solve after move")
  {
    nls::petsc::NonlinearProblem problem(comm);
    problem.set_F(assemble_residual, b);
    problem.set_J([](const Vec x, Mat Jmat, Mat)
                  { assemble_jacobian(x, Jmat); }, J);

    // The SNES callback context points at the problem, so the callbacks
    // must survive a move
    nls::petsc::NonlinearProblem moved(std::move(problem));
    moved.solve(x);
    check_solution(x);

    nls::petsc::NonlinearProblem assigned(comm);
    assigned = std::move(moved);
    VecSet(x, PetscScalar(1));
    assigned.solve(x);
    check_solution(x);
  }

  SECTION("Wrap an existing SNES")
  {
    SNES snes;
    SNESCreate(comm, &snes);
    {
      nls::petsc::NonlinearProblem problem(snes, true);
      problem.set_F(assemble_residual, b);
      problem.set_J([](const Vec x, Mat Jmat, Mat)
                    { assemble_jacobian(x, Jmat); }, J);
      problem.solve(x);
      check_solution(x);
    }

    // The problem held its own reference, so snes is still valid
    PetscObjectReference((PetscObject)snes);
    PetscObjectDereference((PetscObject)snes);
    SNESDestroy(&snes);
  }

  MatDestroy(&J);
  VecDestroy(&b);
  VecDestroy(&x);
}

#endif
