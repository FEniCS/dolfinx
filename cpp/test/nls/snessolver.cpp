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
  CHECK(VecGetArrayRead(x, &_x) == 0);
  PetscScalar* _b;
  CHECK(VecGetArray(b, &_b) == 0);
  std::span<const PetscScalar> xs(_x, local_size);
  std::span<PetscScalar> bs(_b, local_size);
  for (PetscInt i = 0; i < local_size; ++i)
    bs[i] = xs[i] * xs[i] - PetscScalar(2);
  CHECK(VecRestoreArray(b, &_b) == 0);
  CHECK(VecRestoreArrayRead(x, &_x) == 0);
}

// Jacobian df/dx = diag(2 x_i)
void assemble_jacobian(const Vec x, Mat J)
{
  PetscInt r0, r1;
  CHECK(VecGetOwnershipRange(x, &r0, &r1) == 0);
  const PetscScalar* _x;
  CHECK(VecGetArrayRead(x, &_x) == 0);
  std::span<const PetscScalar> xs(_x, local_size);
  CHECK(MatZeroEntries(J) == 0);
  for (PetscInt i = 0; i < local_size; ++i)
  {
    PetscInt row = r0 + i;
    PetscScalar v = PetscScalar(2) * xs[i];
    CHECK(MatSetValues(J, 1, &row, 1, &row, &v, INSERT_VALUES) == 0);
  }
  CHECK(VecRestoreArrayRead(x, &_x) == 0);
  CHECK(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY) == 0);
  CHECK(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY) == 0);
}

// Residual of two decoupled blocks, f(x_i)_j = x_ij^2 - (2 + i), whose
// positive roots are sqrt(2) and sqrt(3)
void assemble_residual_nest(const Vec x, Vec b)
{
  PetscInt n = 0;
  Vec *x_sub = nullptr, *b_sub = nullptr;
  CHECK(VecNestGetSubVecs(x, &n, &x_sub) == 0);
  CHECK(VecNestGetSubVecs(b, &n, &b_sub) == 0);
  for (PetscInt i = 0; i < n; ++i)
  {
    const PetscScalar* _x;
    CHECK(VecGetArrayRead(x_sub[i], &_x) == 0);
    PetscScalar* _b;
    CHECK(VecGetArray(b_sub[i], &_b) == 0);
    std::span<const PetscScalar> xs(_x, local_size);
    std::span<PetscScalar> bs(_b, local_size);
    for (PetscInt j = 0; j < local_size; ++j)
      bs[j] = xs[j] * xs[j] - PetscScalar(2 + i);
    CHECK(VecRestoreArray(b_sub[i], &_b) == 0);
    CHECK(VecRestoreArrayRead(x_sub[i], &_x) == 0);
  }
}

// Block-diagonal Jacobian of the decoupled blocks
void assemble_jacobian_nest(const Vec x, Mat J)
{
  PetscInt n = 0;
  Vec* x_sub = nullptr;
  CHECK(VecNestGetSubVecs(x, &n, &x_sub) == 0);
  for (PetscInt i = 0; i < n; ++i)
  {
    Mat J_sub = nullptr;
    CHECK(MatNestGetSubMat(J, i, i, &J_sub) == 0);
    assemble_jacobian(x_sub[i], J_sub);
  }
  CHECK(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY) == 0);
  CHECK(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY) == 0);
}

// Residual/Jacobian callback matching PETSc's raw SNESFunction /
// SNESJacobianFunction signature, taking the target constant c via ctx
// instead of via a captured/member SNESSolver, to exercise wiring
// SNESSetFunction/SNESSetJacobian directly. noexcept: no invoke() here
// to re-throw after the solve, so terminate cleanly rather than let an
// exception hit PETSc's C frames.
PetscErrorCode residual_ctx(SNES, Vec x, Vec b, void* ctx) noexcept
{
  const PetscScalar c = *static_cast<const PetscScalar*>(ctx);
  const PetscScalar* _x;
  CHECK(VecGetArrayRead(x, &_x) == 0);
  PetscScalar* _b;
  CHECK(VecGetArray(b, &_b) == 0);
  std::span<const PetscScalar> xs(_x, local_size);
  std::span<PetscScalar> bs(_b, local_size);
  for (PetscInt i = 0; i < local_size; ++i)
    bs[i] = xs[i] * xs[i] - c;
  CHECK(VecRestoreArray(b, &_b) == 0);
  CHECK(VecRestoreArrayRead(x, &_x) == 0);
  return PETSC_SUCCESS;
}

PetscErrorCode jacobian_ctx(SNES, Vec x, Mat J, Mat, void*) noexcept
{
  assemble_jacobian(x, J);
  return PETSC_SUCCESS;
}

// Reference count of a PETSc object
template <typename O>
PetscInt ref_count(O obj)
{
  PetscInt count = 0;
  CHECK(PetscObjectGetReference(reinterpret_cast<PetscObject>(obj), &count)
        == 0);
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
  CHECK(SNESGetFunctionNorm(snes, &fnorm) == 0);
  const double round_off
      = 8 * std::numeric_limits<PetscReal>::epsilon() * std::abs(root);
  const double tol = std::clamp(
      static_cast<double>(fnorm) / (2 * std::abs(root)), round_off, 1e-3);

  const PetscScalar* _x;
  CHECK(VecGetArrayRead(x, &_x) == 0);
  std::span<const PetscScalar> xs(_x, local_size);
  for (PetscScalar xi : xs)
  {
    CHECK_THAT(static_cast<double>(PetscRealPart(xi)),
               Catch::Matchers::WithinAbs(root, tol));
  }
  CHECK(VecRestoreArrayRead(x, &_x) == 0);
}
} // namespace

TEST_CASE("Solve nonlinear problem with SNES", "[nls_snes]")
{
  int argc = 0;
  char** argv = nullptr;
  REQUIRE(PetscInitialize(&argc, &argv, nullptr, nullptr) == 0);

  MPI_Comm comm = MPI_COMM_WORLD;
  Vec x, b;
  CHECK(VecCreateMPI(comm, local_size, PETSC_DETERMINE, &x) == 0);
  CHECK(VecDuplicate(x, &b) == 0);
  CHECK(VecSet(x, PetscScalar(1)) == 0);

  Mat J;
  CHECK(MatCreateAIJ(comm, local_size, local_size, PETSC_DETERMINE,
                     PETSC_DETERMINE, 1, nullptr, 0, nullptr, &J)
        == 0);

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

    CHECK(solver.solve(x) > 0);
    PetscInt num_it = 0;
    CHECK(SNESGetIterationNumber(solver.snes(), &num_it) == 0);
    CHECK(num_it > 0);
    check_solution(x, solver.snes());
  }

  SECTION("Manually wire SNESSetFunction/SNESSetJacobian with a raw context")
  {
    // SNESSolver used only for SNES creation/destruction; set_F/set_J are
    // never called, so the residual/Jacobian callbacks and their context
    // are wired directly, bypassing SNESSolver's own trampolines
    nls::petsc::SNESSolver solver(comm);

    PetscScalar c = 2;
    CHECK(SNESSetFunction(solver.snes(), b, residual_ctx, &c) == 0);
    CHECK(SNESSetJacobian(solver.snes(), J, J, jacobian_ctx, &c) == 0);

    // Confirm the context PETSc will invoke the callbacks with is exactly
    // &c, not SNESSolver's own application-context pointer
    void* fctx = nullptr;
    CHECK(SNESGetFunction(solver.snes(), nullptr, nullptr, &fctx) == 0);
    CHECK(fctx == &c);
    void* jctx = nullptr;
    CHECK(SNESGetJacobian(solver.snes(), nullptr, nullptr, nullptr, &jctx)
          == 0);
    CHECK(jctx == &c);

    CHECK(solver.solve(x) > 0);
    PetscInt num_it = 0;
    CHECK(SNESGetIterationNumber(solver.snes(), &num_it) == 0);
    CHECK(num_it > 0);
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
    CHECK(moved.solve(x) > 0);
    check_solution(x, moved.snes());
    PetscInt num_it = 0;
    CHECK(SNESGetIterationNumber(moved.snes(), &num_it) == 0);
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

    CHECK(VecSet(x, PetscScalar(1)) == 0);
    steps.clear();
    CHECK(assigned.solve(x) > 0);
    check_solution(x, assigned.snes());
    CHECK(SNESGetIterationNumber(assigned.snes(), &num_it) == 0);
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
    CHECK(VecSet(x, PetscScalar(-1)) == 0);
    CHECK(solver.solve(x) > 0);
    check_solution(x, solver.snes(), -std::sqrt(2.0));
  }

  SECTION("Repeated solves")
  {
    nls::petsc::SNESSolver solver(comm);
    solver.set_F(assemble_residual, b);
    solver.set_J([](const Vec x, Mat Jmat, Mat) { assemble_jacobian(x, Jmat); },
                 J);

    CHECK(solver.solve(x) > 0);
    check_solution(x, solver.snes());

    // A solver that has converged can be solved again
    CHECK(VecSet(x, PetscScalar(5)) == 0);
    CHECK(solver.solve(x) > 0);
    PetscInt num_it = 0;
    CHECK(SNESGetIterationNumber(solver.snes(), &num_it) == 0);
    CHECK(num_it > 0);
    check_solution(x, solver.snes());
  }

  SECTION("Separate preconditioner matrix")
  {
    Mat P;
    CHECK(MatCreateAIJ(comm, local_size, local_size, PETSC_DETERMINE,
                       PETSC_DETERMINE, 1, nullptr, 0, nullptr, &P)
          == 0);

    nls::petsc::SNESSolver solver(comm);
    solver.set_F(assemble_residual, b);
    solver.set_J(
        [](const Vec x, Mat Jmat, Mat Pmat)
        {
          assemble_jacobian(x, Jmat);
          assemble_jacobian(x, Pmat);
        },
        J, P);
    CHECK(solver.solve(x) > 0);
    check_solution(x, solver.snes());

    Mat Jmat_snes, Pmat_snes;
    CHECK(
        SNESGetJacobian(solver.snes(), &Jmat_snes, &Pmat_snes, nullptr, nullptr)
        == 0);
    CHECK(Jmat_snes == J);
    CHECK(Pmat_snes == P);

    // The preconditioner matrix was assembled into, not left at zero
    PetscReal norm;
    CHECK(MatNorm(P, NORM_FROBENIUS, &norm) == 0);
    CHECK(norm > 0.0);

    CHECK(MatDestroy(&P) == 0);
  }

  SECTION("Update hook")
  {
    nls::petsc::SNESSolver solver(comm);
    solver.set_F(assemble_residual, b);
    solver.set_J([](const Vec x, Mat Jmat, Mat) { assemble_jacobian(x, Jmat); },
                 J);

    std::vector<PetscInt> steps;
    solver.set_update([&steps](PetscInt step) { steps.push_back(step); });

    CHECK(solver.solve(x) > 0);
    check_solution(x, solver.snes());

    // The hook runs once at the start of each iteration
    PetscInt num_it = 0;
    CHECK(SNESGetIterationNumber(solver.snes(), &num_it) == 0);
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
    CHECK(VecDuplicate(x, &x_local) == 0);
    CHECK(VecDuplicate(b, &b_local) == 0);
    CHECK(VecSet(x_local, PetscScalar(1)) == 0);

    nls::petsc::SNESSolver solver(comm);
    solver.set_F([](const Vec, Vec)
                 { throw std::runtime_error("Residual failed"); }, b_local);
    solver.set_J([](const Vec x, Mat Jmat, Mat) { assemble_jacobian(x, Jmat); },
                 J);

    // The exception from the callback is re-thrown, not a PETSc error
    try
    {
      static_cast<void>(solver.solve(x_local));
      FAIL("Expected the callback exception to be re-thrown.");
    }
    catch (const std::runtime_error& e)
    {
      CHECK(std::string(e.what()) == "Residual failed");
    }

    CHECK(VecDestroy(&b_local) == 0);
    CHECK(VecDestroy(&x_local) == 0);
  }

  SECTION("Exception in the Jacobian callback")
  {
    // As above, the solver and its vectors are dead after the throw
    Vec x_local, b_local;
    CHECK(VecDuplicate(x, &x_local) == 0);
    CHECK(VecDuplicate(b, &b_local) == 0);
    CHECK(VecSet(x_local, PetscScalar(1)) == 0);

    nls::petsc::SNESSolver solver(comm);
    solver.set_F(assemble_residual, b_local);
    solver.set_J([](const Vec, Mat, Mat)
                 { throw std::runtime_error("Jacobian failed"); }, J);

    try
    {
      static_cast<void>(solver.solve(x_local));
      FAIL("Expected the callback exception to be re-thrown.");
    }
    catch (const std::runtime_error& e)
    {
      CHECK(std::string(e.what()) == "Jacobian failed");
    }

    CHECK(VecDestroy(&b_local) == 0);
    CHECK(VecDestroy(&x_local) == 0);
  }

  SECTION("Exception in the update hook")
  {
    // As above, the solver and its vectors are dead after the throw
    Vec x_local, b_local;
    CHECK(VecDuplicate(x, &x_local) == 0);
    CHECK(VecDuplicate(b, &b_local) == 0);
    CHECK(VecSet(x_local, PetscScalar(1)) == 0);

    nls::petsc::SNESSolver solver(comm);
    solver.set_F(assemble_residual, b_local);
    solver.set_J([](const Vec x, Mat Jmat, Mat) { assemble_jacobian(x, Jmat); },
                 J);
    solver.set_update([](PetscInt)
                      { throw std::runtime_error("Update failed"); });

    try
    {
      static_cast<void>(solver.solve(x_local));
      FAIL("Expected the callback exception to be re-thrown.");
    }
    catch (const std::runtime_error& e)
    {
      CHECK(std::string(e.what()) == "Update failed");
    }

    CHECK(VecDestroy(&b_local) == 0);
    CHECK(VecDestroy(&x_local) == 0);
  }

  SECTION("Non-convergence is not an error")
  {
    CHECK(PetscOptionsSetValue(nullptr, "-max_it_snes_max_it", "1") == 0);

    nls::petsc::SNESSolver solver(comm);
    solver.set_F(assemble_residual, b);
    solver.set_J([](const Vec x, Mat Jmat, Mat) { assemble_jacobian(x, Jmat); },
                 J);
    solver.set_options_prefix("max_it_");
    solver.set_from_options();

    CHECK_NOTHROW(solver.solve(x));
    SNESConvergedReason reason;
    CHECK(SNESGetConvergedReason(solver.snes(), &reason) == 0);
    CHECK(reason == SNES_DIVERGED_MAX_IT);

    CHECK(PetscOptionsClearValue(nullptr, "-max_it_snes_max_it") == 0);
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
    CHECK(SNESSolve(solver.snes(), nullptr, x) == 0);
    check_solution(x, solver.snes());

    PetscInt num_it = 0;
    CHECK(SNESGetIterationNumber(solver.snes(), &num_it) == 0);
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
      CHECK(VecCreateMPI(comm, local_size, PETSC_DETERMINE, &x_sub[i]) == 0);
      CHECK(VecDuplicate(x_sub[i], &b_sub[i]) == 0);
      CHECK(VecSet(x_sub[i], PetscScalar(1)) == 0);
      CHECK(MatCreateAIJ(comm, local_size, local_size, PETSC_DETERMINE,
                         PETSC_DETERMINE, 1, nullptr, 0, nullptr, &J_sub[i])
            == 0);
    }

    Vec x_nest, b_nest;
    CHECK(VecCreateNest(comm, 2, nullptr, x_sub.data(), &x_nest) == 0);
    CHECK(VecCreateNest(comm, 2, nullptr, b_sub.data(), &b_nest) == 0);

    std::array<Mat, 4> blocks{J_sub[0], nullptr, nullptr, J_sub[1]};
    Mat J_nest;
    CHECK(MatCreateNest(comm, 2, nullptr, 2, nullptr, blocks.data(), &J_nest)
          == 0);

    // A nest matrix cannot be factored directly, so precondition the
    // blocks separately. PETSc takes the splits from the nest.
    CHECK(PetscOptionsSetValue(nullptr, "-nest_ksp_type", "gmres") == 0);
    CHECK(PetscOptionsSetValue(nullptr, "-nest_pc_type", "fieldsplit") == 0);
    CHECK(PetscOptionsSetValue(nullptr, "-nest_fieldsplit_ksp_type", "preonly")
          == 0);
    CHECK(PetscOptionsSetValue(nullptr, "-nest_fieldsplit_pc_type", "jacobi")
          == 0);

    // The solver holds Vec and Mat, so nest objects pass through it
    // untouched
    nls::petsc::SNESSolver solver(comm);
    solver.set_F(assemble_residual_nest, b_nest);
    solver.set_J([](const Vec x, Mat Jmat, Mat)
                 { assemble_jacobian_nest(x, Jmat); }, J_nest);
    solver.set_options_prefix("nest_");
    solver.set_from_options();

    CHECK(solver.solve(x_nest) > 0);
    PetscInt num_it = 0;
    CHECK(SNESGetIterationNumber(solver.snes(), &num_it) == 0);
    CHECK(num_it > 0);

    // The solver did not substitute anything for the nest matrix
    Mat Jmat_snes;
    CHECK(SNESGetJacobian(solver.snes(), &Jmat_snes, nullptr, nullptr, nullptr)
          == 0);
    CHECK(Jmat_snes == J_nest);

    // The nest shares its sub-vectors, so the solution is visible there
    check_solution(x_sub[0], solver.snes(), std::sqrt(2.0));
    check_solution(x_sub[1], solver.snes(), std::sqrt(3.0));

    for (const char* opt :
         {"-nest_ksp_type", "-nest_pc_type", "-nest_fieldsplit_ksp_type",
          "-nest_fieldsplit_pc_type"})
    {
      CHECK(PetscOptionsClearValue(nullptr, opt) == 0);
    }

    CHECK(MatDestroy(&J_nest) == 0);
    CHECK(VecDestroy(&b_nest) == 0);
    CHECK(VecDestroy(&x_nest) == 0);
    for (std::size_t i = 0; i < 2; ++i)
    {
      CHECK(MatDestroy(&J_sub[i]) == 0);
      CHECK(VecDestroy(&b_sub[i]) == 0);
      CHECK(VecDestroy(&x_sub[i]) == 0);
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
    CHECK(MatCreateAIJ(comm, local_size, local_size, PETSC_DETERMINE,
                       PETSC_DETERMINE, 1, nullptr, 0, nullptr, &P)
          == 0);
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

    CHECK(MatDestroy(&P) == 0);
  }

  SECTION("Wrap an existing SNES")
  {
    SNES snes;
    CHECK(SNESCreate(comm, &snes) == 0);
    const PetscInt snes_count = ref_count(snes);
    {
      // +1: inc_ref_count, so the SNES cannot be destroyed while the
      // solver is using it
      nls::petsc::SNESSolver solver(snes, true);
      CHECK(ref_count(snes) == snes_count + 1);
      solver.set_F(assemble_residual, b);
      solver.set_J([](const Vec x, Mat Jmat, Mat)
                   { assemble_jacobian(x, Jmat); }, J);
      CHECK(solver.solve(x) > 0);
      check_solution(x, solver.snes());
    }

    // The solver released its reference, so snes is still valid
    CHECK(ref_count(snes) == snes_count);
    CHECK(SNESDestroy(&snes) == 0);
  }

  CHECK(MatDestroy(&J) == 0);
  CHECK(VecDestroy(&b) == 0);
  CHECK(VecDestroy(&x) == 0);
}

#endif
