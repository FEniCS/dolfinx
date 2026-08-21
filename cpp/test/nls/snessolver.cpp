// Copyright (C) 2026 Jack S. Hale
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later
//
// Unit tests for nls::petsc::SNESSolver

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#ifdef HAS_PETSC

#include <algorithm>
#include <array>
#include <cmath>
#include <dolfinx/nls/SNESSolver.h>
#include <limits>
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

// Residual of two decoupled blocks, f(x_i)_j = x_ij^2 - (2 + i), whose
// positive roots are sqrt(2) and sqrt(3)
void assemble_residual_nest(const Vec x, Vec b)
{
  PetscInt n = 0;
  Vec *x_sub = nullptr, *b_sub = nullptr;
  VecNestGetSubVecs(x, &n, &x_sub);
  VecNestGetSubVecs(b, &n, &b_sub);
  for (PetscInt i = 0; i < n; ++i)
  {
    const PetscScalar* _x;
    VecGetArrayRead(x_sub[i], &_x);
    PetscScalar* _b;
    VecGetArray(b_sub[i], &_b);
    std::span<const PetscScalar> xs(_x, local_size);
    std::span<PetscScalar> bs(_b, local_size);
    for (PetscInt j = 0; j < local_size; ++j)
      bs[j] = xs[j] * xs[j] - PetscScalar(2 + i);
    VecRestoreArray(b_sub[i], &_b);
    VecRestoreArrayRead(x_sub[i], &_x);
  }
}

// Block-diagonal Jacobian of the decoupled blocks
void assemble_jacobian_nest(const Vec x, Mat J)
{
  PetscInt n = 0;
  Vec* x_sub = nullptr;
  VecNestGetSubVecs(x, &n, &x_sub);
  for (PetscInt i = 0; i < n; ++i)
  {
    Mat J_sub = nullptr;
    MatNestGetSubMat(J, i, i, &J_sub);
    assemble_jacobian(x_sub[i], J_sub);
  }
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

// Check that all entries of x are `root`, by default the positive root
// sqrt(2), to the accuracy the solver stopped at rather than to a fixed
// tolerance, which would depend on the precision PETSc was built with
void check_solution(const Vec x, SNES snes, double root = std::sqrt(2.0))
{
  // For f(x) = x^2 - c, |x - root| <= |f(x)| / |f'(root)|, and the
  // residual norm over all entries bounds each |f(x_i)|. Clamped below
  // by round-off, and above so that a solve stopping far from the root
  // fails rather than widening its own tolerance.
  PetscReal fnorm = 0;
  SNESGetFunctionNorm(snes, &fnorm);
  const double round_off
      = 8 * std::numeric_limits<PetscReal>::epsilon() * std::abs(root);
  const double tol = std::clamp(
      static_cast<double>(fnorm) / (2 * std::abs(root)), round_off, 1e-3);

  const PetscScalar* _x;
  VecGetArrayRead(x, &_x);
  std::span<const PetscScalar> xs(_x, local_size);
  for (PetscScalar xi : xs)
  {
    CHECK_THAT(static_cast<double>(PetscRealPart(xi)),
               Catch::Matchers::WithinAbs(root, tol));
  }
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
    nls::petsc::SNESSolver solver(comm);
    solver.set_F(assemble_residual, b);
    solver.set_J([](const Vec x, Mat Jmat, Mat) { assemble_jacobian(x, Jmat); },
                 J);

    // PETSc reports an unset prefix as a null pointer, not ""
    CHECK(solver.get_options_prefix().empty());

    solver.set_options_prefix("test_snes_");
    CHECK(solver.get_options_prefix() == "test_snes_");
    solver.set_from_options();

    PetscInt num_it = solver.solve(x);
    CHECK(num_it > 0);
    SNESConvergedReason reason;
    SNESGetConvergedReason(solver.snes(), &reason);
    CHECK(reason > 0);
    check_solution(x, solver.snes());
  }

  SECTION("Solve after move")
  {
    nls::petsc::SNESSolver solver(comm);
    solver.set_F(assemble_residual, b);
    solver.set_J([](const Vec x, Mat Jmat, Mat) { assemble_jacobian(x, Jmat); },
                 J);

    // The update hook has no context of its own and recovers the
    // solver from the residual callback, so it exercises a second route
    // back to the solver
    std::vector<PetscInt> steps;
    solver.set_update([&steps](PetscInt step) { steps.push_back(step); });

    // The SNES callback context points at the solver, so the callbacks
    // must survive a move
    nls::petsc::SNESSolver moved(std::move(solver));
    PetscInt num_it = moved.solve(x);
    check_solution(x, moved.snes());
    CHECK(steps.size() == std::size_t(num_it));

    nls::petsc::SNESSolver assigned(comm);
    assigned = std::move(moved);

    // Re-register the hook on the moved-to solver, recording into a
    // different vector. A moved-from callable is left in a valid but
    // unspecified state, and remains callable on some implementations,
    // so only a distinct target shows which solver the callback
    // actually reached.
    std::vector<PetscInt> assigned_steps;
    assigned.set_update([&assigned_steps](PetscInt step)
                        { assigned_steps.push_back(step); });

    VecSet(x, PetscScalar(1));
    steps.clear();
    num_it = assigned.solve(x);
    check_solution(x, assigned.snes());
    CHECK(assigned_steps.size() == std::size_t(num_it));
    CHECK(steps.empty());
  }

  SECTION("Initial guess")
  {
    nls::petsc::SNESSolver solver(comm);
    solver.set_F(assemble_residual, b);
    solver.set_J([](const Vec x, Mat Jmat, Mat) { assemble_jacobian(x, Jmat); },
                 J);

    // Newton from a negative guess converges to the negative root, so
    // the vector passed to solve is used as the starting point
    VecSet(x, PetscScalar(-1));
    solver.solve(x);
    check_solution(x, solver.snes(), -std::sqrt(2.0));
  }

  SECTION("Repeated solves")
  {
    nls::petsc::SNESSolver solver(comm);
    solver.set_F(assemble_residual, b);
    solver.set_J([](const Vec x, Mat Jmat, Mat) { assemble_jacobian(x, Jmat); },
                 J);

    solver.solve(x);
    check_solution(x, solver.snes());

    // A solver that has converged can be solved again
    VecSet(x, PetscScalar(5));
    PetscInt num_it = solver.solve(x);
    CHECK(num_it > 0);
    check_solution(x, solver.snes());
  }

  SECTION("Separate preconditioner matrix")
  {
    Mat P;
    MatCreateAIJ(comm, local_size, local_size, PETSC_DETERMINE, PETSC_DETERMINE,
                 1, nullptr, 0, nullptr, &P);

    nls::petsc::SNESSolver solver(comm);
    solver.set_F(assemble_residual, b);
    solver.set_J(
        [](const Vec x, Mat Jmat, Mat Pmat)
        {
          assemble_jacobian(x, Jmat);
          assemble_jacobian(x, Pmat);
        },
        J, P);
    solver.solve(x);
    check_solution(x, solver.snes());

    Mat Jmat_snes, Pmat_snes;
    SNESGetJacobian(solver.snes(), &Jmat_snes, &Pmat_snes, nullptr, nullptr);
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
    nls::petsc::SNESSolver solver(comm);
    solver.set_F(assemble_residual, b);
    solver.set_J([](const Vec x, Mat Jmat, Mat) { assemble_jacobian(x, Jmat); },
                 J);

    std::vector<PetscInt> steps;
    solver.set_update([&steps](PetscInt step) { steps.push_back(step); });

    PetscInt num_it = solver.solve(x);
    check_solution(x, solver.snes());

    // The hook runs once at the start of each iteration
    REQUIRE(steps.size() == std::size_t(num_it));
    for (std::size_t i = 0; i < steps.size(); ++i)
      CHECK(steps[i] == PetscInt(i));
  }

  SECTION("Exception in a callback")
  {
    // PETSc does not restore its state when an aborted solve unwinds,
    // so the solver and the vectors it holds are dead afterwards. Use
    // vectors local to this section.
    Vec x_local, b_local;
    VecDuplicate(x, &x_local);
    VecDuplicate(b, &b_local);
    VecSet(x_local, PetscScalar(1));

    nls::petsc::SNESSolver solver(comm);
    solver.set_F([](const Vec, Vec)
                 { throw std::runtime_error("Residual failed"); }, b_local);
    solver.set_J([](const Vec x, Mat Jmat, Mat) { assemble_jacobian(x, Jmat); },
                 J);

    // The exception from the callback is re-thrown, not a PETSc error
    try
    {
      solver.solve(x_local);
      FAIL("Expected the callback exception to be re-thrown.");
    }
    catch (const std::runtime_error& e)
    {
      CHECK(std::string(e.what()) == "Residual failed");
    }

    VecDestroy(&b_local);
    VecDestroy(&x_local);
  }

  SECTION("Exception in the Jacobian callback")
  {
    // As above, the solver and its vectors are dead after the throw
    Vec x_local, b_local;
    VecDuplicate(x, &x_local);
    VecDuplicate(b, &b_local);
    VecSet(x_local, PetscScalar(1));

    nls::petsc::SNESSolver solver(comm);
    solver.set_F(assemble_residual, b_local);
    solver.set_J([](const Vec, Mat, Mat)
                 { throw std::runtime_error("Jacobian failed"); }, J);

    try
    {
      solver.solve(x_local);
      FAIL("Expected the callback exception to be re-thrown.");
    }
    catch (const std::runtime_error& e)
    {
      CHECK(std::string(e.what()) == "Jacobian failed");
    }

    VecDestroy(&b_local);
    VecDestroy(&x_local);
  }

  SECTION("Exception in the update hook")
  {
    // As above, the solver and its vectors are dead after the throw
    Vec x_local, b_local;
    VecDuplicate(x, &x_local);
    VecDuplicate(b, &b_local);
    VecSet(x_local, PetscScalar(1));

    nls::petsc::SNESSolver solver(comm);
    solver.set_F(assemble_residual, b_local);
    solver.set_J([](const Vec x, Mat Jmat, Mat) { assemble_jacobian(x, Jmat); },
                 J);
    solver.set_update([](PetscInt)
                      { throw std::runtime_error("Update failed"); });

    try
    {
      solver.solve(x_local);
      FAIL("Expected the callback exception to be re-thrown.");
    }
    catch (const std::runtime_error& e)
    {
      CHECK(std::string(e.what()) == "Update failed");
    }

    VecDestroy(&b_local);
    VecDestroy(&x_local);
  }

  SECTION("Non-convergence is not an error")
  {
    PetscOptionsSetValue(nullptr, "-max_it_snes_max_it", "1");

    nls::petsc::SNESSolver solver(comm);
    solver.set_F(assemble_residual, b);
    solver.set_J([](const Vec x, Mat Jmat, Mat) { assemble_jacobian(x, Jmat); },
                 J);
    solver.set_options_prefix("max_it_");
    solver.set_from_options();

    CHECK_NOTHROW(solver.solve(x));
    SNESConvergedReason reason;
    SNESGetConvergedReason(solver.snes(), &reason);
    CHECK(reason == SNES_DIVERGED_MAX_IT);

    PetscOptionsClearValue(nullptr, "-max_it_snes_max_it");
  }

  SECTION("Solve through the SNES object")
  {
    nls::petsc::SNESSolver solver(comm);
    solver.set_F(assemble_residual, b);
    solver.set_J([](const Vec x, Mat Jmat, Mat) { assemble_jacobian(x, Jmat); },
                 J);

    std::vector<PetscInt> steps;
    solver.set_update([&steps](PetscInt step) { steps.push_back(step); });

    // Callbacks recover the solver from the SNES, so bypassing solve()
    // works, including the update hook
    SNESSolve(solver.snes(), nullptr, x);
    check_solution(x, solver.snes());

    PetscInt num_it = 0;
    SNESGetIterationNumber(solver.snes(), &num_it);
    CHECK(steps.size() == std::size_t(num_it));
  }

  SECTION("Nest matrices and vectors")
  {
    // Two decoupled blocks with different roots, so that a mix-up
    // between them would show up in the solution
    std::array<Vec, 2> x_sub, b_sub;
    std::array<Mat, 2> J_sub;
    for (std::size_t i = 0; i < 2; ++i)
    {
      VecCreateMPI(comm, local_size, PETSC_DETERMINE, &x_sub[i]);
      VecDuplicate(x_sub[i], &b_sub[i]);
      VecSet(x_sub[i], PetscScalar(1));
      MatCreateAIJ(comm, local_size, local_size, PETSC_DETERMINE,
                   PETSC_DETERMINE, 1, nullptr, 0, nullptr, &J_sub[i]);
    }

    Vec x_nest, b_nest;
    VecCreateNest(comm, 2, nullptr, x_sub.data(), &x_nest);
    VecCreateNest(comm, 2, nullptr, b_sub.data(), &b_nest);

    std::array<Mat, 4> blocks{J_sub[0], nullptr, nullptr, J_sub[1]};
    Mat J_nest;
    MatCreateNest(comm, 2, nullptr, 2, nullptr, blocks.data(), &J_nest);

    // A nest matrix cannot be factored directly, so precondition the
    // blocks separately. PETSc takes the splits from the nest.
    PetscOptionsSetValue(nullptr, "-nest_ksp_type", "gmres");
    PetscOptionsSetValue(nullptr, "-nest_pc_type", "fieldsplit");
    PetscOptionsSetValue(nullptr, "-nest_fieldsplit_ksp_type", "preonly");
    PetscOptionsSetValue(nullptr, "-nest_fieldsplit_pc_type", "jacobi");

    // The solver holds Vec and Mat, so nest objects pass through it
    // untouched
    nls::petsc::SNESSolver solver(comm);
    solver.set_F(assemble_residual_nest, b_nest);
    solver.set_J([](const Vec x, Mat Jmat, Mat)
                 { assemble_jacobian_nest(x, Jmat); }, J_nest);
    solver.set_options_prefix("nest_");
    solver.set_from_options();

    PetscInt num_it = solver.solve(x_nest);
    CHECK(num_it > 0);

    // The solver did not substitute anything for the nest matrix
    Mat Jmat_snes;
    SNESGetJacobian(solver.snes(), &Jmat_snes, nullptr, nullptr, nullptr);
    CHECK(Jmat_snes == J_nest);
    SNESConvergedReason reason;
    SNESGetConvergedReason(solver.snes(), &reason);
    CHECK(reason > 0);

    // The nest shares its sub-vectors, so the solution is visible there
    check_solution(x_sub[0], solver.snes(), std::sqrt(2.0));
    check_solution(x_sub[1], solver.snes(), std::sqrt(3.0));

    for (const char* opt :
         {"-nest_ksp_type", "-nest_pc_type", "-nest_fieldsplit_ksp_type",
          "-nest_fieldsplit_pc_type"})
    {
      PetscOptionsClearValue(nullptr, opt);
    }

    MatDestroy(&J_nest);
    VecDestroy(&b_nest);
    VecDestroy(&x_nest);
    for (std::size_t i = 0; i < 2; ++i)
    {
      MatDestroy(&J_sub[i]);
      VecDestroy(&b_sub[i]);
      VecDestroy(&x_sub[i]);
    }
  }

  SECTION("Reference counting")
  {
    const PetscInt b_count = ref_count(b), J_count = ref_count(J);
    {
      nls::petsc::SNESSolver solver(comm);
      solver.set_F(assemble_residual, b);

      // +2: set_F references the vector, and SNESSetFunction takes a
      // reference of its own
      CHECK(ref_count(b) == b_count + 2);

      // Unchanged: set_F references the new vector before releasing the
      // one it held, so passing the same vector twice is a no-op rather
      // than a release
      solver.set_F(assemble_residual, b);
      CHECK(ref_count(b) == b_count + 2);

      // +4: with no preconditioner matrix given, the Jacobian is used as
      // its own preconditioner, so it is passed twice to set_J and twice
      // to SNESSetJacobian, each of which references both arguments
      solver.set_J([](const Vec x, Mat Jmat, Mat)
                   { assemble_jacobian(x, Jmat); }, J);
      CHECK(ref_count(J) == J_count + 4);
    }

    // Back to baseline: the destructor releases the references the
    // solver took, and SNESDestroy those the SNES took
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
      nls::petsc::SNESSolver solver(comm);
      // +2 each: the two matrices are distinct, so each is referenced
      // once by the solver and once by the SNES
      solver.set_J([](const Vec x, Mat Jmat, Mat)
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
      // +1: inc_ref_count, so the SNES cannot be destroyed while the
      // solver is using it
      nls::petsc::SNESSolver solver(snes, true);
      CHECK(ref_count(snes) == snes_count + 1);
      solver.set_F(assemble_residual, b);
      solver.set_J([](const Vec x, Mat Jmat, Mat)
                   { assemble_jacobian(x, Jmat); }, J);
      solver.solve(x);
      check_solution(x, solver.snes());
    }

    // The solver released its reference, so snes is still valid
    CHECK(ref_count(snes) == snes_count);
    SNESDestroy(&snes);
  }

  MatDestroy(&J);
  VecDestroy(&b);
  VecDestroy(&x);
}

#endif
