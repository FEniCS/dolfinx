// Copyright (C) 2004-2018 Johan Hoffman, Johan Jansson, Anders Logg and
// Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "PETScMatrix.h"
#include "PETScVector.h"
#include "Vector.h"
#include "utils.h"
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/common/log.h>
#include <dolfinx/la/SparsityPattern.h>
#include <iostream>
#include <sstream>

using namespace dolfinx;
using namespace dolfinx::la;

//-----------------------------------------------------------------------------
Mat la::petsc::create_matrix(MPI_Comm comm,
                             const dolfinx::la::SparsityPattern& sp,
                             const std::string& type)
{
  PetscErrorCode ierr;
  Mat A;
  ierr = MatCreate(comm, &A);
  if (ierr != 0)
    petsc::error(ierr, __FILE__, "MatCreate");

  // Get IndexMaps from sparsity patterm, and block size
  std::array maps = {sp.index_map(0), sp.index_map(1)};
  const std::array bs = {sp.block_size(0), sp.block_size(1)};

  if (!type.empty())
    MatSetType(A, type.c_str());

  // Get global and local dimensions
  const std::int64_t M = bs[0] * maps[0]->size_global();
  const std::int64_t N = bs[1] * maps[1]->size_global();
  const std::int32_t m = bs[0] * maps[0]->size_local();
  const std::int32_t n = bs[1] * maps[1]->size_local();

  // Set matrix size
  ierr = MatSetSizes(A, m, n, M, N);
  if (ierr != 0)
    petsc::error(ierr, __FILE__, "MatSetSizes");

  // Get number of nonzeros for each row from sparsity pattern
  const graph::AdjacencyList<std::int32_t>& diagonal_pattern
      = sp.diagonal_pattern();
  const graph::AdjacencyList<std::int32_t>& off_diagonal_pattern
      = sp.off_diagonal_pattern();

  // Apply PETSc options from the options database to the matrix (this
  // includes changing the matrix type to one specified by the user)
  ierr = MatSetFromOptions(A);
  if (ierr != 0)
    petsc::error(ierr, __FILE__, "MatSetFromOptions");

  // Find a common block size across rows/columns
  const int _bs = (bs[0] == bs[1] ? bs[0] : 1);

  // Build data to initialise sparsity pattern (modify for block size)
  std::vector<PetscInt> _nnz_diag, _nnz_offdiag;
  if (bs[0] == bs[1])
  {
    _nnz_diag.resize(maps[0]->size_local());
    _nnz_offdiag.resize(maps[0]->size_local());
    for (std::size_t i = 0; i < _nnz_diag.size(); ++i)
      _nnz_diag[i] = diagonal_pattern.links(i).size();
    for (std::size_t i = 0; i < _nnz_offdiag.size(); ++i)
      _nnz_offdiag[i] = off_diagonal_pattern.links(i).size();
  }
  else
  {
    // Expand for block size 1
    _nnz_diag.resize(maps[0]->size_local() * bs[0]);
    _nnz_offdiag.resize(maps[0]->size_local() * bs[0]);
    for (std::size_t i = 0; i < _nnz_diag.size(); ++i)
      _nnz_diag[i] = bs[1] * diagonal_pattern.links(i / bs[0]).size();
    for (std::size_t i = 0; i < _nnz_offdiag.size(); ++i)
      _nnz_offdiag[i] = bs[1] * off_diagonal_pattern.links(i / bs[0]).size();
  }

  // Allocate space for matrix
  ierr = MatXAIJSetPreallocation(A, _bs, _nnz_diag.data(), _nnz_offdiag.data(),
                                 nullptr, nullptr);
  if (ierr != 0)
    petsc::error(ierr, __FILE__, "MatXIJSetPreallocation");

  // Set block sizes
  ierr = MatSetBlockSizes(A, bs[0], bs[1]);
  if (ierr != 0)
    petsc::error(ierr, __FILE__, "MatSetBlockSizes");

  // Create PETSc local-to-global map/index sets
  ISLocalToGlobalMapping local_to_global0;
  const std::vector map0 = maps[0]->global_indices();
  const std::vector<PetscInt> _map0(map0.begin(), map0.end());
  ierr = ISLocalToGlobalMappingCreate(MPI_COMM_SELF, bs[0], _map0.size(),
                                      _map0.data(), PETSC_COPY_VALUES,
                                      &local_to_global0);

  if (ierr != 0)
    petsc::error(ierr, __FILE__, "ISLocalToGlobalMappingCreate");

  // Check for common index maps
  if (maps[0] == maps[1] and bs[0] == bs[1])
  {
    ierr = MatSetLocalToGlobalMapping(A, local_to_global0, local_to_global0);
    if (ierr != 0)
      petsc::error(ierr, __FILE__, "MatSetLocalToGlobalMapping");
  }
  else
  {
    ISLocalToGlobalMapping local_to_global1;
    const std::vector map1 = maps[1]->global_indices();
    const std::vector<PetscInt> _map1(map1.begin(), map1.end());
    ierr = ISLocalToGlobalMappingCreate(MPI_COMM_SELF, bs[1], _map1.size(),
                                        _map1.data(), PETSC_COPY_VALUES,
                                        &local_to_global1);
    if (ierr != 0)
      petsc::error(ierr, __FILE__, "ISLocalToGlobalMappingCreate");
    ierr = MatSetLocalToGlobalMapping(A, local_to_global0, local_to_global1);
    if (ierr != 0)
      petsc::error(ierr, __FILE__, "MatSetLocalToGlobalMapping");
    ierr = ISLocalToGlobalMappingDestroy(&local_to_global1);
    if (ierr != 0)
      petsc::error(ierr, __FILE__, "ISLocalToGlobalMappingDestroy");
  }

  // Clean up local-to-global 0
  ierr = ISLocalToGlobalMappingDestroy(&local_to_global0);
  if (ierr != 0)
    petsc::error(ierr, __FILE__, "ISLocalToGlobalMappingDestroy");

  // Note: This should be called after having set the local-to-global
  // map for MATIS (this is a dummy call if A is not of type MATIS)
  // ierr = MatISSetPreallocation(A, 0, _nnz_diag.data(), 0,
  // _nnz_offdiag.data()); if (ierr != 0)
  //   error(ierr, __FILE__, "MatISSetPreallocation");

  // Set some options on Mat object
  ierr = MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE);
  if (ierr != 0)
    petsc::error(ierr, __FILE__, "MatSetOption");
  ierr = MatSetOption(A, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE);
  if (ierr != 0)
    petsc::error(ierr, __FILE__, "MatSetOption");

  return A;
}
//-----------------------------------------------------------------------------
MatNullSpace la::petsc::create_nullspace(MPI_Comm comm,
                                         const xtl::span<const Vec>& basis)
{
  MatNullSpace ns = nullptr;
  PetscErrorCode ierr
      = MatNullSpaceCreate(comm, PETSC_FALSE, basis.size(), basis.data(), &ns);
  if (ierr != 0)
    petsc::error(ierr, __FILE__, "MatNullSpaceCreate");
  return ns;
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
void petsc::Options::set(std::string option)
{
  petsc::Options::set<std::string>(option, "");
}
//-----------------------------------------------------------------------------
void petsc::Options::clear(std::string option)
{
  if (option[0] != '-')
    option = '-' + option;

  PetscErrorCode ierr;
  ierr = PetscOptionsClearValue(nullptr, option.c_str());
  if (ierr != 0)
    petsc::error(ierr, __FILE__, "PetscOptionsClearValue");
}
//-----------------------------------------------------------------------------
void petsc::Options::clear()
{
  PetscErrorCode ierr = PetscOptionsClear(nullptr);
  if (ierr != 0)
    petsc::error(ierr, __FILE__, "PetscOptionsClear");
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
petsc::Operator::Operator(Mat A, bool inc_ref_count) : _matA(A)
{
  assert(A);
  if (inc_ref_count)
    PetscObjectReference((PetscObject)_matA);
}
//-----------------------------------------------------------------------------
petsc::Operator::Operator(Operator&& A) : _matA(std::exchange(A._matA, nullptr))
{
}
//-----------------------------------------------------------------------------
petsc::Operator::~Operator()
{
  // Decrease reference count (PETSc will destroy object once reference
  // counts reached zero)
  if (_matA)
    MatDestroy(&_matA);
}
//-----------------------------------------------------------------------------
petsc::Operator& petsc::Operator::operator=(Operator&& A)
{
  std::swap(_matA, A._matA);
  return *this;
}
//-----------------------------------------------------------------------------
std::array<std::int64_t, 2> petsc::Operator::size() const
{
  assert(_matA);
  PetscInt m(0), n(0);
  PetscErrorCode ierr = MatGetSize(_matA, &m, &n);
  if (ierr != 0)
    petsc::error(ierr, __FILE__, "MetGetSize");
  return {{m, n}};
}
//-----------------------------------------------------------------------------
petsc::Vector petsc::Operator::create_vector(std::size_t dim) const
{
  assert(_matA);
  PetscErrorCode ierr;

  Vec x = nullptr;
  if (dim == 0)
  {
    ierr = MatCreateVecs(_matA, nullptr, &x);
    if (ierr != 0)
      petsc::error(ierr, __FILE__, "MatCreateVecs");
  }
  else if (dim == 1)
  {
    ierr = MatCreateVecs(_matA, &x, nullptr);
    if (ierr != 0)
      petsc::error(ierr, __FILE__, "MatCreateVecs");
  }
  else
  {
    LOG(ERROR) << "Cannot initialize PETSc vector to match PETSc matrix. "
               << "Dimension must be 0 or 1, not " << dim;
    throw std::runtime_error("Invalid dimension");
  }

  return Vector(x, false);
}
//-----------------------------------------------------------------------------
Mat petsc::Operator::mat() const { return _matA; }
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                  const std::int32_t*, const PetscScalar*)>
petsc::Matrix::set_fn(Mat A, InsertMode mode)
{
  return [A, mode, cache = std::vector<PetscInt>()](
             std::int32_t m, const std::int32_t* rows, std::int32_t n,
             const std::int32_t* cols, const PetscScalar* vals) mutable -> int
  {
    PetscErrorCode ierr;
#ifdef PETSC_USE_64BIT_INDICES
    cache.resize(m + n);
    std::copy_n(rows, m, cache.begin());
    std::copy_n(cols, n, std::next(cache.begin(), m));
    const PetscInt *_rows = cache.data(), *_cols = _rows + m;
    ierr = MatSetValuesLocal(A, m, _rows, n, _cols, vals, mode);
#else
    ierr = MatSetValuesLocal(A, m, rows, n, cols, vals, mode);
#endif

#ifdef DEBUG
    if (ierr != 0)
      petsc::error(ierr, __FILE__, "MatSetValuesLocal");
#endif

    return ierr;
  };
}
//-----------------------------------------------------------------------------
std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                  const std::int32_t*, const PetscScalar*)>
petsc::Matrix::set_block_fn(Mat A, InsertMode mode)
{
  return [A, mode, cache = std::vector<PetscInt>()](
             std::int32_t m, const std::int32_t* rows, std::int32_t n,
             const std::int32_t* cols, const PetscScalar* vals) mutable -> int
  {
    PetscErrorCode ierr;
#ifdef PETSC_USE_64BIT_INDICES
    cache.resize(m + n);
    std::copy_n(rows, m, cache.begin());
    std::copy_n(cols, n, std::next(cache.begin(), m));
    const PetscInt *_rows = cache.data(), *_cols = _rows + m;
    ierr = MatSetValuesBlockedLocal(A, m, _rows, n, _cols, vals, mode);
#else
    ierr = MatSetValuesBlockedLocal(A, m, rows, n, cols, vals, mode);
#endif

#ifdef DEBUG
    if (ierr != 0)
      petsc::error(ierr, __FILE__, "MatSetValuesBlockedLocal");
#endif

    return ierr;
  };
}
//-----------------------------------------------------------------------------
std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                  const std::int32_t*, const PetscScalar*)>
petsc::Matrix::set_block_expand_fn(Mat A, int bs0, int bs1, InsertMode mode)
{
  if (bs0 == 1 and bs1 == 1)
    return set_fn(A, mode);

  return [A, bs0, bs1, mode, cache0 = std::vector<PetscInt>(),
          cache1 = std::vector<PetscInt>()](
             std::int32_t m, const std::int32_t* rows, std::int32_t n,
             const std::int32_t* cols, const PetscScalar* vals) mutable -> int
  {
    PetscErrorCode ierr;
    cache0.resize(bs0 * m);
    cache1.resize(bs1 * n);
    for (std::int32_t i = 0; i < m; ++i)
      for (int k = 0; k < bs0; ++k)
        cache0[bs0 * i + k] = bs0 * rows[i] + k;
    for (std::int32_t i = 0; i < n; ++i)
      for (int k = 0; k < bs1; ++k)
        cache1[bs1 * i + k] = bs1 * cols[i] + k;

    ierr = MatSetValuesLocal(A, cache0.size(), cache0.data(), cache1.size(),
                             cache1.data(), vals, mode);
#ifdef DEBUG
    if (ierr != 0)
      petsc::error(ierr, __FILE__, "MatSetValuesLocal");
#endif
    return ierr;
  };
}
//-----------------------------------------------------------------------------
petsc::Matrix::Matrix(MPI_Comm comm, const SparsityPattern& sp,
                      const std::string& type)
    : Operator(petsc::create_matrix(comm, sp, type), false)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
petsc::Matrix::Matrix(Mat A, bool inc_ref_count) : Operator(A, inc_ref_count)
{
  // Reference count to A is incremented in base class
}
//-----------------------------------------------------------------------------
double petsc::Matrix::norm(Norm norm_type) const
{
  assert(_matA);
  PetscErrorCode ierr;
  double value = 0.0;
  switch (norm_type)
  {
  case Norm::l1:
    ierr = MatNorm(_matA, NORM_1, &value);
    break;
  case Norm::linf:
    ierr = MatNorm(_matA, NORM_INFINITY, &value);
    break;
  case Norm::frobenius:
    ierr = MatNorm(_matA, NORM_FROBENIUS, &value);
    break;
  default:
    throw std::runtime_error("Unknown PETSc Mat norm type");
  }

  if (ierr != 0)
    petsc::error(ierr, __FILE__, "MatNorm");

  return value;
}
//-----------------------------------------------------------------------------
void petsc::Matrix::apply(AssemblyType type)
{
  common::Timer timer("Apply (PETScMatrix)");

  assert(_matA);
  PetscErrorCode ierr;

  MatAssemblyType petsc_type = MAT_FINAL_ASSEMBLY;
  if (type == AssemblyType::FLUSH)
    petsc_type = MAT_FLUSH_ASSEMBLY;

  ierr = MatAssemblyBegin(_matA, petsc_type);
  if (ierr != 0)
    petsc::error(ierr, __FILE__, "MatAssemblyBegin");
  ierr = MatAssemblyEnd(_matA, petsc_type);
  if (ierr != 0)
    petsc::error(ierr, __FILE__, "MatAssemblyEnd");
}
//-----------------------------------------------------------------------------
void petsc::Matrix::set_options_prefix(std::string options_prefix)
{
  assert(_matA);
  MatSetOptionsPrefix(_matA, options_prefix.c_str());
}
//-----------------------------------------------------------------------------
std::string petsc::Matrix::get_options_prefix() const
{
  assert(_matA);
  const char* prefix = nullptr;
  MatGetOptionsPrefix(_matA, &prefix);
  return std::string(prefix);
}
//-----------------------------------------------------------------------------
void petsc::Matrix::set_from_options()
{
  assert(_matA);
  MatSetFromOptions(_matA);
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
petsc::KrylovSolver::KrylovSolver(MPI_Comm comm) : _ksp(nullptr)
{
  PetscErrorCode ierr;

  // Create PETSc KSP object
  ierr = KSPCreate(comm, &_ksp);
  if (ierr != 0)
    petsc::error(ierr, __FILE__, "KSPCreate");
}
//-----------------------------------------------------------------------------
petsc::KrylovSolver::KrylovSolver(KSP ksp, bool inc_ref_count) : _ksp(ksp)
{
  assert(_ksp);
  if (inc_ref_count)
  {
    PetscErrorCode ierr = PetscObjectReference((PetscObject)_ksp);
    if (ierr != 0)
      petsc::error(ierr, __FILE__, "PetscObjectReference");
  }
}
//-----------------------------------------------------------------------------
petsc::KrylovSolver::KrylovSolver(KrylovSolver&& solver)
    : _ksp(std::exchange(solver._ksp, nullptr))
{
  // Do nothing
}
//-----------------------------------------------------------------------------
petsc::KrylovSolver::~KrylovSolver()
{
  if (_ksp)
    KSPDestroy(&_ksp);
}
//-----------------------------------------------------------------------------
petsc::KrylovSolver& petsc::KrylovSolver::operator=(KrylovSolver&& solver)
{
  std::swap(_ksp, solver._ksp);
  return *this;
}
//-----------------------------------------------------------------------------
void petsc::KrylovSolver::set_operator(const Mat A) { set_operators(A, A); }
//-----------------------------------------------------------------------------
void petsc::KrylovSolver::set_operators(const Mat A, const Mat P)
{
  assert(A);
  assert(_ksp);
  PetscErrorCode ierr;
  ierr = KSPSetOperators(_ksp, A, P);
  if (ierr != 0)
    petsc::error(ierr, __FILE__, "KSPSetOperators");
}
//-----------------------------------------------------------------------------
int petsc::KrylovSolver::solve(Vec x, const Vec b, bool transpose) const
{
  common::Timer timer("PETSc Krylov solver");
  assert(x);
  assert(b);

  // Get PETSc operators
  Mat _A, _P;
  KSPGetOperators(_ksp, &_A, &_P);
  assert(_A);

  // Create wrapper around PETSc Mat object
  // la::PETScOperator A(_A);

  PetscErrorCode ierr;

  // // Check dimensions
  // const std::array<std::int64_t, 2> size = A.size();
  // if (size[0] != b.size())
  // {
  //   log::dolfin_error(
  //       "PETScKrylovSolver.cpp",
  //       "unable to solve linear system with PETSc Krylov solver",
  //       "Non-matching dimensions for linear system (matrix has %ld "
  //       "rows and right-hand side vector has %ld rows)",
  //       size[0], b.size());
  // }

  // Solve linear system
  LOG(INFO) << "PETSc Krylov solver starting to solve system.";

  // Solve system
  if (!transpose)
  {
    ierr = KSPSolve(_ksp, b, x);
    if (ierr != 0)
      petsc::error(ierr, __FILE__, "KSPSolve");
  }
  else
  {
    ierr = KSPSolveTranspose(_ksp, b, x);
    if (ierr != 0)
      petsc::error(ierr, __FILE__, "KSPSolve");
  }

  // FIXME: Remove ghost updating?
  // Update ghost values in solution vector
  Vec xg;
  VecGhostGetLocalForm(x, &xg);
  const bool is_ghosted = xg ? true : false;
  VecGhostRestoreLocalForm(x, &xg);
  if (is_ghosted)
  {
    VecGhostUpdateBegin(x, INSERT_VALUES, SCATTER_FORWARD);
    VecGhostUpdateEnd(x, INSERT_VALUES, SCATTER_FORWARD);
  }

  // Get the number of iterations
  PetscInt num_iterations = 0;
  ierr = KSPGetIterationNumber(_ksp, &num_iterations);
  if (ierr != 0)
    petsc::error(ierr, __FILE__, "KSPGetIterationNumber");

  // Check if the solution converged and print error/warning if not
  // converged
  KSPConvergedReason reason;
  ierr = KSPGetConvergedReason(_ksp, &reason);
  if (ierr != 0)
    petsc::error(ierr, __FILE__, "KSPGetConvergedReason");
  if (reason < 0)
  {
    /*
    // Get solver residual norm
    double rnorm = 0.0;
    ierr = KSPGetResidualNorm(_ksp, &rnorm);
    if (ierr != 0) error(ierr, __FILE__, "KSPGetResidualNorm");
    const char *reason_str = KSPConvergedReasons[reason];
    bool error_on_nonconvergence =
    this->parameters["error_on_nonconvergence"].is_set() ?
    this->parameters["error_on_nonconvergence"] : true;
    if (error_on_nonconvergence)
    {
      log::dolfin_error("PETScKrylovSolver.cpp",
                   "solve linear system using PETSc Krylov solver",
                   "Solution failed to converge in %i iterations (PETSc reason
    %s, residual norm ||r|| = %e)",
                   static_cast<int>(num_iterations), reason_str, rnorm);
    }
    else
    {
      log::warning("Krylov solver did not converge in %i iterations (PETSc
    reason %s,
    residual norm ||r|| = %e).",
              num_iterations, reason_str, rnorm);
    }
    */
  }

  // Report results
  // if (report && dolfinx::MPI::rank(this->comm()) == 0)
  //  write_report(num_iterations, reason);

  return num_iterations;
}
//-----------------------------------------------------------------------------
void petsc::KrylovSolver::set_options_prefix(std::string options_prefix)
{
  // Set options prefix
  assert(_ksp);
  PetscErrorCode ierr = KSPSetOptionsPrefix(_ksp, options_prefix.c_str());
  if (ierr != 0)
    petsc::error(ierr, __FILE__, "KSPSetOptionsPrefix");
}
//-----------------------------------------------------------------------------
std::string petsc::KrylovSolver::get_options_prefix() const
{
  assert(_ksp);
  const char* prefix = nullptr;
  PetscErrorCode ierr = KSPGetOptionsPrefix(_ksp, &prefix);
  if (ierr != 0)
    petsc::error(ierr, __FILE__, "KSPGetOptionsPrefix");
  return std::string(prefix);
}
//-----------------------------------------------------------------------------
void petsc::KrylovSolver::set_from_options() const
{
  assert(_ksp);
  PetscErrorCode ierr = KSPSetFromOptions(_ksp);
  if (ierr != 0)
    petsc::error(ierr, __FILE__, "KSPSetFromOptions");
}
//-----------------------------------------------------------------------------
KSP petsc::KrylovSolver::ksp() const { return _ksp; }
//-----------------------------------------------------------------------------
