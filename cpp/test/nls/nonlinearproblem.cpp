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
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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

// Reference count of a PETSc object
template <typename O>
PetscInt ref_count(O obj)
{
  PetscInt count = 0;
  PetscObjectGetReference(reinterpret_cast<PetscObject>(obj), &count);
  return count;
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

  SECTION("Separate preconditioner matrix")
  {
    Mat P;
    MatCreateAIJ(comm, local_size, local_size, PETSC_DETERMINE, PETSC_DETERMINE,
                 1, nullptr, 0, nullptr, &P);

    nls::petsc::NonlinearProblem problem(comm);
    problem.set_F(assemble_residual, b);
    problem.set_J(
        [](const Vec x, Mat Jmat, Mat Pmat)
        {
          assemble_jacobian(x, Jmat);
          assemble_jacobian(x, Pmat);
        },
        J, P);
    problem.solve(x);
    check_solution(x);

    Mat Jmat_snes, Pmat_snes;
    SNESGetJacobian(problem.snes(), &Jmat_snes, &Pmat_snes, nullptr, nullptr);
    CHECK(Jmat_snes == J);
    CHECK(Pmat_snes == P);

    // The preconditioner matrix was assembled into, not left at zero
    PetscReal norm;
    MatNorm(P, NORM_FROBENIUS, &norm);
    CHECK(norm > 0.0);

    MatDestroy(&P);
  }

  SECTION("Update hook")
  {
    nls::petsc::NonlinearProblem problem(comm);
    problem.set_F(assemble_residual, b);
    problem.set_J([](const Vec x, Mat Jmat, Mat)
                  { assemble_jacobian(x, Jmat); }, J);

    std::vector<int> steps;
    problem.set_update([&steps](int step) { steps.push_back(step); });

    int num_it = problem.solve(x);
    check_solution(x);

    // The hook runs once at the start of each iteration
    REQUIRE(steps.size() == std::size_t(num_it));
    for (std::size_t i = 0; i < steps.size(); ++i)
      CHECK(steps[i] == int(i));
  }

  SECTION("Exception in a callback")
  {
    // PETSc does not restore its state when an aborted solve unwinds,
    // so the problem and the vectors it holds are dead afterwards. Use
    // vectors local to this section.
    Vec x_local, b_local;
    VecDuplicate(x, &x_local);
    VecDuplicate(b, &b_local);
    VecSet(x_local, PetscScalar(1));

    nls::petsc::NonlinearProblem problem(comm);
    problem.set_F([](const Vec, Vec)
                  { throw std::runtime_error("Residual failed"); }, b_local);
    problem.set_J([](const Vec x, Mat Jmat, Mat)
                  { assemble_jacobian(x, Jmat); }, J);

    // The exception from the callback is re-thrown, not a PETSc error
    try
    {
      problem.solve(x_local);
      FAIL("Expected the callback exception to be re-thrown.");
    }
    catch (const std::runtime_error& e)
    {
      CHECK(std::string(e.what()) == "Residual failed");
    }

    VecDestroy(&b_local);
    VecDestroy(&x_local);
  }

  SECTION("Non-convergence is not an error")
  {
    PetscOptionsSetValue(nullptr, "-max_it_snes_max_it", "1");

    nls::petsc::NonlinearProblem problem(comm);
    problem.set_F(assemble_residual, b);
    problem.set_J([](const Vec x, Mat Jmat, Mat)
                  { assemble_jacobian(x, Jmat); }, J);
    problem.set_options_prefix("max_it_");
    problem.set_from_options();

    CHECK_NOTHROW(problem.solve(x));
    SNESConvergedReason reason;
    SNESGetConvergedReason(problem.snes(), &reason);
    CHECK(reason == SNES_DIVERGED_MAX_IT);

    PetscOptionsClearValue(nullptr, "-max_it_snes_max_it");
  }

  SECTION("Reference counting")
  {
    const PetscInt b_count = ref_count(b), J_count = ref_count(J);
    {
      nls::petsc::NonlinearProblem problem(comm);
      problem.set_F(assemble_residual, b);

      // One reference held by the problem, one by the SNES
      CHECK(ref_count(b) == b_count + 2);

      // Re-setting with the same vector must not accumulate references
      problem.set_F(assemble_residual, b);
      CHECK(ref_count(b) == b_count + 2);

      // With no preconditioner matrix the Jacobian is used for both, so
      // it is referenced twice by the problem and twice by the SNES
      problem.set_J([](const Vec x, Mat Jmat, Mat)
                    { assemble_jacobian(x, Jmat); }, J);
      CHECK(ref_count(J) == J_count + 4);
    }

    // Destroying the problem releases every reference it took
    CHECK(ref_count(b) == b_count);
    CHECK(ref_count(J) == J_count);
  }

  SECTION("Reference counting with a preconditioner matrix")
  {
    Mat P;
    MatCreateAIJ(comm, local_size, local_size, PETSC_DETERMINE, PETSC_DETERMINE,
                 1, nullptr, 0, nullptr, &P);
    const PetscInt J_count = ref_count(J), P_count = ref_count(P);
    {
      nls::petsc::NonlinearProblem problem(comm);
      problem.set_J([](const Vec x, Mat Jmat, Mat)
                    { assemble_jacobian(x, Jmat); }, J, P);
      CHECK(ref_count(J) == J_count + 2);
      CHECK(ref_count(P) == P_count + 2);
    }
    CHECK(ref_count(J) == J_count);
    CHECK(ref_count(P) == P_count);

    MatDestroy(&P);
  }

  SECTION("Wrap an existing SNES")
  {
    SNES snes;
    SNESCreate(comm, &snes);
    const PetscInt snes_count = ref_count(snes);
    {
      nls::petsc::NonlinearProblem problem(snes, true);
      CHECK(ref_count(snes) == snes_count + 1);
      problem.set_F(assemble_residual, b);
      problem.set_J([](const Vec x, Mat Jmat, Mat)
                    { assemble_jacobian(x, Jmat); }, J);
      problem.solve(x);
      check_solution(x);
    }

    // The problem released its reference, so snes is still valid
    CHECK(ref_count(snes) == snes_count);
    SNESDestroy(&snes);
  }

  MatDestroy(&J);
  VecDestroy(&b);
  VecDestroy(&x);
}

#endif
