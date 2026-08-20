// Copyright (C) 2026 Jack S. Hale
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later
//
// Unit tests for la::SLEPcEigenSolver

#ifdef HAS_SLEPC

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <complex>
#include <dolfinx/la/slepc.h>
#include <limits>
#include <petscmat.h>
#include <slepceps.h>
#include <stdexcept>
#include <utility>

using namespace dolfinx;

namespace
{
/// Relative tolerance on computed eigenvalues. Both test matrices are
/// normal, so the eigenvalue error is bounded by the residual the
/// solver stops at, which is looser in single precision.
constexpr double eig_rtol
    = std::numeric_limits<PetscReal>::epsilon() < 1e-10 ? 1e-8 : 1e-5;

/// Create a distributed diagonal matrix with entries 1, 2, ..., N, so
/// that the spectrum is known exactly.
Mat create_diagonal_matrix(PetscInt N)
{
  Mat A = nullptr;
  MatCreate(MPI_COMM_WORLD, &A);
  MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, N, N);
  MatSetType(A, MATAIJ);
  MatSetUp(A);

  PetscInt r0 = 0, r1 = 0;
  MatGetOwnershipRange(A, &r0, &r1);
  for (PetscInt i = r0; i < r1; ++i)
  {
    PetscScalar v = static_cast<PetscReal>(i + 1);
    MatSetValues(A, 1, &i, 1, &i, &v, INSERT_VALUES);
  }

  MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
  return A;
}

/// Create a real non-symmetric block-diagonal matrix built from 2x2
/// blocks [[a, -a], [a, a]] for a = 1, ..., num_blocks. Block a has the
/// complex conjugate eigenvalue pair a(1 +/- i), so the spectrum is
/// known exactly and the pair of largest magnitude is num_blocks(1 +/- i).
Mat create_conjugate_pair_matrix(PetscInt num_blocks)
{
  const PetscInt N = 2 * num_blocks;
  Mat A = nullptr;
  MatCreate(MPI_COMM_WORLD, &A);
  MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, N, N);
  MatSetType(A, MATAIJ);
  MatSeqAIJSetPreallocation(A, 2, nullptr);
  MatMPIAIJSetPreallocation(A, 2, nullptr, 2, nullptr);

  PetscInt r0 = 0, r1 = 0;
  MatGetOwnershipRange(A, &r0, &r1);
  for (PetscInt i = r0; i < r1; ++i)
  {
    const PetscInt block = i / 2;
    const PetscScalar a = static_cast<PetscReal>(block + 1);
    if (i % 2 == 0)
    {
      const PetscInt cols[2] = {i, i + 1};
      const PetscScalar vals[2] = {a, -a};
      MatSetValues(A, 1, &i, 2, cols, vals, INSERT_VALUES);
    }
    else
    {
      const PetscInt cols[2] = {i - 1, i};
      const PetscScalar vals[2] = {a, a};
      MatSetValues(A, 1, &i, 2, cols, vals, INSERT_VALUES);
    }
  }

  MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
  return A;
}
} // namespace

TEST_CASE("SLEPc eigenvalue solver", "[slepc]")
{
  int argc = 0;
  char** argv = nullptr;
  REQUIRE(SlepcInitialize(&argc, &argv, nullptr, nullptr) == 0);

  constexpr PetscInt N = 20;

  SECTION("Solve a standard Hermitian eigenvalue problem")
  {
    Mat A = create_diagonal_matrix(N);
    la::SLEPcEigenSolver solver(MPI_COMM_WORLD);
    solver.set_operators(A, nullptr);

    // Diagonal matrix is Hermitian; ask for the largest eigenvalues,
    // which Krylov methods recover first
    EPSSetProblemType(solver.eps(), EPS_HEP);
    EPSSetWhichEigenpairs(solver.eps(), EPS_LARGEST_MAGNITUDE);
    EPSSetDimensions(solver.eps(), 3, PETSC_DETERMINE, PETSC_DETERMINE);

    solver.solve();

    const PetscInt nconv = solver.get_number_converged();
    REQUIRE(nconv >= 3);

    // Spectrum is {1, ..., N}, so the three largest are N, N-1, N-2
    for (PetscInt i = 0; i < 3; ++i)
    {
      std::complex<PetscReal> l = solver.get_eigenvalue(i);
      CHECK_THAT(
          static_cast<double>(l.real()),
          Catch::Matchers::WithinRel(static_cast<double>(N - i), eig_rtol));
      CHECK_THAT(static_cast<double>(l.imag()),
                 Catch::Matchers::WithinAbs(0.0, 1e-12));
    }

    SECTION("Eigenpair matches eigenvalue")
    {
      Vec r = nullptr, c = nullptr;
      MatCreateVecs(A, nullptr, &r);
      MatCreateVecs(A, nullptr, &c);

      PetscScalar lr = 0, lc = 0;
      solver.get_eigenpair(lr, lc, r, c, 0);
      CHECK_THAT(static_cast<double>(PetscRealPart(lr)),
                 Catch::Matchers::WithinRel(static_cast<double>(N), eig_rtol));

      VecDestroy(&r);
      VecDestroy(&c);
    }

    SECTION("Out-of-range indices throw rather than return garbage")
    {
      PetscScalar lr = 0, lc = 0;
      CHECK_THROWS_AS(solver.get_eigenvalue(-1), std::runtime_error);
      CHECK_THROWS_AS(solver.get_eigenvalue(nconv), std::runtime_error);
      CHECK_THROWS_AS(solver.get_eigenpair(lr, lc, nullptr, nullptr, -1),
                      std::runtime_error);
      CHECK_THROWS_AS(solver.get_eigenpair(lr, lc, nullptr, nullptr, nconv),
                      std::runtime_error);
    }

    MatDestroy(&A);
  }

  SECTION("Complex conjugate eigenpair of a non-symmetric operator")
  {
    constexpr PetscInt num_blocks = 3;
    Mat A = create_conjugate_pair_matrix(num_blocks);
    la::SLEPcEigenSolver solver(MPI_COMM_WORLD);
    solver.set_operators(A, nullptr);

    EPSSetProblemType(solver.eps(), EPS_NHEP);
    EPSSetWhichEigenpairs(solver.eps(), EPS_LARGEST_MAGNITUDE);
    EPSSetDimensions(solver.eps(), 2, PETSC_DETERMINE, PETSC_DETERMINE);

    solver.solve();
    REQUIRE(solver.get_number_converged() >= 2);

    // get_eigenvalue is independent of the PETSc scalar type: the pair
    // of largest magnitude is num_blocks(1 +/- i) in both real and
    // complex builds
    constexpr double expected = static_cast<double>(num_blocks);
    std::complex<PetscReal> l0 = solver.get_eigenvalue(0);
    std::complex<PetscReal> l1 = solver.get_eigenvalue(1);
    for (std::complex<PetscReal> l : {l0, l1})
    {
      CHECK_THAT(static_cast<double>(l.real()),
                 Catch::Matchers::WithinRel(expected, eig_rtol));
      CHECK_THAT(std::abs(static_cast<double>(l.imag())),
                 Catch::Matchers::WithinRel(expected, eig_rtol));
    }

    // A conjugate pair, so the imaginary parts have opposite signs
    CHECK(l0.imag() * l1.imag() < 0);

    SECTION("Raw eigenpair storage follows the PETSc scalar type")
    {
      PetscScalar lr = 0, lc = 0;
      solver.get_eigenpair(lr, lc, nullptr, nullptr, 0);
#ifdef PETSC_USE_COMPLEX
      // Eigenvalue held entirely in lr, and lc set to zero
      CHECK_THAT(std::abs(static_cast<double>(PetscImaginaryPart(lr))),
                 Catch::Matchers::WithinRel(expected, eig_rtol));
      CHECK_THAT(static_cast<double>(PetscRealPart(lc)),
                 Catch::Matchers::WithinAbs(0.0, 1e-12));
      CHECK_THAT(static_cast<double>(PetscImaginaryPart(lc)),
                 Catch::Matchers::WithinAbs(0.0, 1e-12));
#else
      // Real scalars split the pair across lr and lc
      CHECK_THAT(static_cast<double>(lr),
                 Catch::Matchers::WithinRel(expected, eig_rtol));
      CHECK_THAT(std::abs(static_cast<double>(lc)),
                 Catch::Matchers::WithinRel(expected, eig_rtol));
#endif
    }

    MatDestroy(&A);
  }

  SECTION("Solving before setting operators throws")
  {
    // Collective-safe: every rank evaluates the same condition and
    // throws before entering any collective
    la::SLEPcEigenSolver solver(MPI_COMM_WORLD);
    CHECK_THROWS_AS(solver.solve(), std::runtime_error);
  }

  SECTION("Options prefix round-trips and is empty when unset")
  {
    la::SLEPcEigenSolver solver(MPI_COMM_WORLD);

    // Must not dereference a null prefix
    CHECK(solver.get_options_prefix().empty());

    solver.set_options_prefix("mysolver_");
    CHECK(solver.get_options_prefix() == "mysolver_");
  }

  SECTION("Wrapping a null EPS throws")
  {
    CHECK_THROWS_AS(la::SLEPcEigenSolver(nullptr, true), std::runtime_error);
  }

  SECTION("Move leaves the source safe to destroy")
  {
    la::SLEPcEigenSolver a(MPI_COMM_WORLD);
    EPS raw = a.eps();
    la::SLEPcEigenSolver b(std::move(a));
    CHECK(b.eps() == raw);
    CHECK(a.eps() == nullptr);
  }
}

#endif
