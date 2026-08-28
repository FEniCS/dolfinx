// Copyright (C) 2004-2026 Johan Hoffman, Johan Jansson, Anders Logg,
// Garth N. Wells and Jack S. Hale
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#ifdef HAS_PETSC

#include "petsc.h"
#include "SparsityPattern.h"
#include "Vector.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/common/log.h>
#include <numeric>
#include <ranges>
#include <stdexcept>
#include <utility>

using namespace dolfinx;
using namespace dolfinx::la;

//-----------------------------------------------------------------------------
std::vector<Vec>
la::petsc::create_vectors(MPI_Comm comm,
                          const std::vector<std::span<const PetscScalar>>& x)
{
  std::vector<Vec> v(x.size());
  for (std::size_t i = 0; i < v.size(); ++i)
  {
    common::petsc::check(
        VecCreateMPI(comm, x[i].size(), PETSC_DETERMINE, &v[i]),
        "VecCreateMPI");
    PetscScalar* data;
    common::petsc::check(VecGetArray(v[i], &data), "VecGetArray");
    std::ranges::copy(x[i], data);
    common::petsc::check(VecRestoreArray(v[i], &data), "VecRestoreArray");
  }

  return v;
}
//-----------------------------------------------------------------------------
Vec la::petsc::create_vector(const common::IndexMap& map, int bs)
{
  return la::petsc::create_vector(map.comm(), map.local_range(), map.ghosts(),
                                  bs);
}
//-----------------------------------------------------------------------------
Vec la::petsc::create_vector(MPI_Comm comm, std::array<std::int64_t, 2> range,
                             std::span<const std::int64_t> ghosts, int bs)
{
  // Get local size
  assert(range[1] >= range[0]);
  std::int32_t local_size = range[1] - range[0];

  Vec x = nullptr;
  std::vector<PetscInt> _ghosts(ghosts.begin(), ghosts.end());
  if (bs == 1)
  {
    common::petsc::check(VecCreateGhost(comm, local_size, PETSC_DETERMINE,
                                        _ghosts.size(), _ghosts.data(), &x),
                         "VecCreateGhost");
  }
  else
  {
    common::petsc::check(VecCreateGhostBlock(comm, bs, bs * local_size,
                                             PETSC_DETERMINE, _ghosts.size(),
                                             _ghosts.data(), &x),
                         "VecCreateGhostBlock");
  }

  assert(x);
  return x;
}
//-----------------------------------------------------------------------------
Vec la::petsc::create_vector_wrap(const common::IndexMap& map, int bs,
                                  std::span<const PetscScalar> x)
{
  const std::int32_t size_local = bs * map.size_local();
  const std::int64_t size_global = bs * map.size_global();
  const std::vector<PetscInt> ghosts(map.ghosts().begin(), map.ghosts().end());
  if (x.size() < static_cast<std::size_t>(size_local) + bs * ghosts.size())
  {
    throw std::runtime_error(
        "Array size is too small for the index map, including ghosts.");
  }

  Vec vec;
  if (bs == 1)
  {
    common::petsc::check(VecCreateGhostWithArray(map.comm(), size_local,
                                                 size_global, ghosts.size(),
                                                 ghosts.data(), x.data(), &vec),
                         "VecCreateGhostWithArray");
  }
  else
  {
    common::petsc::check(VecCreateGhostBlockWithArray(
                             map.comm(), bs, size_local, size_global,
                             ghosts.size(), ghosts.data(), x.data(), &vec),
                         "VecCreateGhostBlockWithArray");
  }

  assert(vec);
  return vec;
}
//-----------------------------------------------------------------------------
std::vector<IS> la::petsc::create_index_sets(
    const std::vector<
        std::pair<std::reference_wrapper<const common::IndexMap>, int>>& maps)
{
  std::vector<IS> is;
  std::int64_t offset = 0;
  for (auto& [map, bs] : maps)
  {
    std::int32_t size = map.get().size_local() + map.get().num_ghosts();
    IS _is;
    common::petsc::check(
        ISCreateStride(PETSC_COMM_SELF, bs * size, offset, 1, &_is),
        "ISCreateStride");
    is.push_back(_is);
    offset += bs * size;
  }

  return is;
}
//-----------------------------------------------------------------------------
std::vector<std::vector<PetscScalar>> la::petsc::get_local_vectors(
    const Vec x,
    const std::vector<
        std::pair<std::reference_wrapper<const common::IndexMap>, int>>& maps)
{
  // Get ghost offset
  int offset_owned = 0;
  for (auto& [map, bs] : maps)
    offset_owned += map.get().size_local() * bs;

  // Unwrap PETSc vector
  Vec x_local;
  common::petsc::check(VecGhostGetLocalForm(x, &x_local),
                       "VecGhostGetLocalForm");
  PetscInt n = 0;
  common::petsc::check(VecGetSize(x_local, &n), "VecGetSize");
  const PetscScalar* array = nullptr;
  common::petsc::check(VecGetArrayRead(x_local, &array), "VecGetArrayRead");
  std::span _x(array, n);

  // Copy PETSc Vec data in to local vectors
  std::vector<std::vector<PetscScalar>> x_b;
  int offset = 0;
  int offset_ghost = offset_owned; // Ghost DoFs start after owned
  for (auto& [map, bs] : maps)
  {
    const std::int32_t size_owned = map.get().size_local() * bs;
    const std::int32_t size_ghost = map.get().num_ghosts() * bs;

    x_b.emplace_back(size_owned + size_ghost);
    std::copy_n(std::next(_x.begin(), offset), size_owned, x_b.back().begin());
    std::copy_n(std::next(_x.begin(), offset_ghost), size_ghost,
                std::next(x_b.back().begin(), size_owned));

    offset += size_owned;
    offset_ghost += size_ghost;
  }

  common::petsc::check(VecRestoreArrayRead(x_local, &array),
                       "VecRestoreArrayRead");
  common::petsc::check(VecGhostRestoreLocalForm(x, &x_local),
                       "VecGhostRestoreLocalForm");

  return x_b;
}
//-----------------------------------------------------------------------------
void la::petsc::scatter_local_vectors(
    Vec x, const std::vector<std::span<const PetscScalar>>& x_b,
    const std::vector<
        std::pair<std::reference_wrapper<const common::IndexMap>, int>>& maps)
{
  if (x_b.size() != maps.size())
    throw std::runtime_error("Mismatch in vector/map size.");

  // Get ghost offset
  int offset_owned = 0;
  for (auto& [map, bs] : maps)
    offset_owned += map.get().size_local() * bs;

  Vec x_local;
  common::petsc::check(VecGhostGetLocalForm(x, &x_local),
                       "VecGhostGetLocalForm");
  PetscInt n = 0;
  common::petsc::check(VecGetSize(x_local, &n), "VecGetSize");
  PetscScalar* array = nullptr;
  common::petsc::check(VecGetArray(x_local, &array), "VecGetArray");
  std::span _x(array, n);

  // Copy local vectors into PETSc Vec
  int offset = 0;
  int offset_ghost = offset_owned; // Ghost DoFs start after owned
  for (std::size_t i = 0; i < maps.size(); ++i)
  {
    const auto& [map, bs] = maps[i];
    std::int32_t size_owned = map.get().size_local() * bs;
    std::copy_n(x_b[i].begin(), size_owned, std::next(_x.begin(), offset));

    std::int32_t size_ghost = map.get().num_ghosts() * bs;
    std::copy_n(std::next(x_b[i].begin(), size_owned), size_ghost,
                std::next(_x.begin(), offset_ghost));

    offset += size_owned;
    offset_ghost += size_ghost;
  }

  common::petsc::check(VecRestoreArray(x_local, &array), "VecRestoreArray");
  common::petsc::check(VecGhostRestoreLocalForm(x, &x_local),
                       "VecGhostRestoreLocalForm");
}
//-----------------------------------------------------------------------------
Mat la::petsc::create_matrix(MPI_Comm comm, const SparsityPattern& sp,
                             std::optional<std::string_view> type)
{
  Mat A;
  common::petsc::check(MatCreate(comm, &A), "MatCreate");

  // Get IndexMaps from sparsity pattern, and block size
  std::array maps = {sp.index_map(0), sp.index_map(1)};
  const std::array bs = {sp.block_size(0), sp.block_size(1)};

  if (type and !type->empty())
    common::petsc::check(MatSetType(A, std::string(*type).c_str()),
                         "MatSetType");

  // Get global and local dimensions
  const std::int64_t M = bs[0] * maps[0]->size_global();
  const std::int64_t N = bs[1] * maps[1]->size_global();
  const std::int32_t m = bs[0] * maps[0]->size_local();
  const std::int32_t n = bs[1] * maps[1]->size_local();

  // Set matrix size
  common::petsc::check(MatSetSizes(A, m, n, M, N), "MatSetSizes");

  // Apply PETSc options from the options database to the matrix (this
  // includes changing the matrix type to one specified by the user)
  common::petsc::check(MatSetFromOptions(A), "MatSetFromOptions");

  // Find a common block size across rows/columns
  const int _bs = (bs[0] == bs[1] ? bs[0] : 1);

  // Build data to initialise sparsity pattern (modify for block size)
  std::vector<PetscInt> _nnz_diag, _nnz_offdiag;
  if (bs[0] == bs[1])
  {
    const std::int32_t size_local = maps[0]->size_local();
    _nnz_diag.resize(size_local);
    _nnz_offdiag.resize(size_local);
    auto rows = std::views::iota(std::int32_t(0), size_local);
    std::ranges::transform(rows, _nnz_diag.begin(),
                           [&sp](std::int32_t i) { return sp.nnz_diag(i); });
    std::ranges::transform(rows, _nnz_offdiag.begin(), [&sp](std::int32_t i)
                           { return sp.nnz_off_diag(i); });
  }
  else
  {
    // Expand for block size 1
    const std::int32_t n = maps[0]->size_local() * bs[0];
    _nnz_diag.resize(n);
    _nnz_offdiag.resize(n);
    auto rows = std::views::iota(std::int32_t(0), n);
    std::ranges::transform(rows, _nnz_diag.begin(), [&sp, &bs](std::int32_t i)
                           { return bs[1] * sp.nnz_diag(i / bs[0]); });
    std::ranges::transform(rows, _nnz_offdiag.begin(),
                           [&sp, &bs](std::int32_t i)
                           { return bs[1] * sp.nnz_off_diag(i / bs[0]); });
  }

  // Allocate space for matrix
  common::petsc::check(MatXAIJSetPreallocation(A, _bs, _nnz_diag.data(),
                                               _nnz_offdiag.data(), nullptr,
                                               nullptr),
                       "MatXAIJSetPreallocation");

  // Set block sizes
  common::petsc::check(MatSetBlockSizes(A, bs[0], bs[1]), "MatSetBlockSizes");

  // Build a PETSc (PetscInt) local-to-global map directly from an
  // IndexMap's local range and ghosts, rather than going via
  // IndexMap::global_indices() (which materialises an intermediate
  // std::int64_t array that would then need a second, full-size
  // conversion pass -- wasteful for the large local sizes seen in
  // practice)
  auto build_l2g = [](const common::IndexMap& map) -> std::vector<PetscInt>
  {
    const std::int32_t size_local = map.size_local();
    std::vector<PetscInt> l2g(size_local + map.num_ghosts());
    std::iota(l2g.begin(), std::next(l2g.begin(), size_local),
              static_cast<PetscInt>(map.local_range()[0]));
    std::ranges::copy(map.ghosts(), std::next(l2g.begin(), size_local));
    return l2g;
  };

  // Create PETSc local-to-global map/index sets
  ISLocalToGlobalMapping local_to_global0;
  std::vector<PetscInt> _map0 = build_l2g(*maps[0]);
  common::petsc::check(ISLocalToGlobalMappingCreate(
                           MPI_COMM_SELF, bs[0], _map0.size(), _map0.data(),
                           PETSC_COPY_VALUES, &local_to_global0),
                       "ISLocalToGlobalMappingCreate");

  // Check for common index maps
  if (maps[0] == maps[1] and bs[0] == bs[1])
  {
    common::petsc::check(
        MatSetLocalToGlobalMapping(A, local_to_global0, local_to_global0),
        "MatSetLocalToGlobalMapping");
  }
  else
  {
    ISLocalToGlobalMapping local_to_global1;
    std::vector<PetscInt> _map1 = build_l2g(*maps[1]);
    common::petsc::check(ISLocalToGlobalMappingCreate(
                             MPI_COMM_SELF, bs[1], _map1.size(), _map1.data(),
                             PETSC_COPY_VALUES, &local_to_global1),
                         "ISLocalToGlobalMappingCreate");
    common::petsc::check(
        MatSetLocalToGlobalMapping(A, local_to_global0, local_to_global1),
        "MatSetLocalToGlobalMapping");
    common::petsc::check(ISLocalToGlobalMappingDestroy(&local_to_global1),
                         "ISLocalToGlobalMappingDestroy");
  }

  // Clean up local-to-global 0
  common::petsc::check(ISLocalToGlobalMappingDestroy(&local_to_global0),
                       "ISLocalToGlobalMappingDestroy");

  // Note: This should be called after having set the local-to-global
  // map for MATIS (this is a dummy call if A is not of type MATIS)
  // ierr = MatISSetPreallocation(A, 0, _nnz_diag.data(), 0,
  // _nnz_offdiag.data()); if (ierr != 0)
  //   error(ierr, __FILE__, "MatISSetPreallocation");

  // Set some options on Mat object
  common::petsc::check(
      MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE),
      "MatSetOption");
  common::petsc::check(MatSetOption(A, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE),
                       "MatSetOption");

  return A;
}
//-----------------------------------------------------------------------------
MatNullSpace la::petsc::create_nullspace(MPI_Comm comm,
                                         std::span<const Vec> basis)
{
  MatNullSpace ns = nullptr;
  common::petsc::check(
      MatNullSpaceCreate(comm, PETSC_FALSE, basis.size(), basis.data(), &ns),
      "MatNullSpaceCreate");
  return ns;
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
petsc::Vector::Vector(const common::IndexMap& map, int bs)
    : _x(la::petsc::create_vector(map, bs))
{
  // Do nothing
}
//-----------------------------------------------------------------------------
petsc::Vector::Vector(Vec x, bool inc_ref_count) : _x(x)
{
  if (!_x)
    throw std::runtime_error("PETSc Vec must be initialised before wrapping");

  if (inc_ref_count)
  {
    common::petsc::check(PetscObjectReference((PetscObject)_x),
                         "PetscObjectReference");
  }
}
//-----------------------------------------------------------------------------
petsc::Vector::Vector(Vector&& v) noexcept : _x(std::exchange(v._x, nullptr)) {}
//-----------------------------------------------------------------------------
petsc::Vector::~Vector()
{
  // Destructor is implicitly noexcept, so a thrown error here calls
  // std::terminate rather than propagating
  if (_x)
    common::petsc::check(VecDestroy(&_x), "VecDestroy");
}
//-----------------------------------------------------------------------------
petsc::Vector& petsc::Vector::operator=(Vector&& v) noexcept
{
  std::swap(_x, v._x);
  return *this;
}
//-----------------------------------------------------------------------------
petsc::Vector petsc::Vector::copy() const
{
  Vec _y;
  common::petsc::check(VecDuplicate(_x, &_y), "VecDuplicate");
  common::petsc::check(VecCopy(_x, _y), "VecCopy");
  return Vector(_y, false);
}
//-----------------------------------------------------------------------------
std::int64_t petsc::Vector::size() const
{
  assert(_x);
  PetscInt n = 0;
  common::petsc::check(VecGetSize(_x, &n), "VecGetSize");
  return n;
}
//-----------------------------------------------------------------------------
std::int32_t petsc::Vector::local_size() const
{
  assert(_x);
  PetscInt n = 0;
  common::petsc::check(VecGetLocalSize(_x, &n), "VecGetLocalSize");
  return n;
}
//-----------------------------------------------------------------------------
std::array<std::int64_t, 2> petsc::Vector::local_range() const
{
  assert(_x);
  PetscInt n0, n1;
  common::petsc::check(VecGetOwnershipRange(_x, &n0, &n1),
                       "VecGetOwnershipRange");
  assert(n0 <= n1);
  return {n0, n1};
}
//-----------------------------------------------------------------------------
MPI_Comm petsc::Vector::comm() const
{
  assert(_x);
  MPI_Comm mpi_comm = MPI_COMM_NULL;
  common::petsc::check(PetscObjectGetComm((PetscObject)(_x), &mpi_comm),
                       "PetscObjectGetComm");
  return mpi_comm;
}
//-----------------------------------------------------------------------------
void petsc::Vector::set_options_prefix(std::string_view options_prefix)
{
  assert(_x);
  common::petsc::check(
      VecSetOptionsPrefix(_x, std::string(options_prefix).c_str()),
      "VecSetOptionsPrefix");
}
//-----------------------------------------------------------------------------
std::string petsc::Vector::get_options_prefix() const
{
  assert(_x);
  const char* prefix = nullptr;
  common::petsc::check(VecGetOptionsPrefix(_x, &prefix), "VecGetOptionsPrefix");
  return prefix ? std::string(prefix) : std::string();
}
//-----------------------------------------------------------------------------
void petsc::Vector::set_from_options()
{
  assert(_x);
  common::petsc::check(VecSetFromOptions(_x), "VecSetFromOptions");
}
//-----------------------------------------------------------------------------
Vec petsc::Vector::vec() const { return _x; }
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
petsc::Matrix::Matrix(MPI_Comm comm, const SparsityPattern& sp,
                      std::optional<std::string_view> type)
    : _matA(petsc::create_matrix(comm, sp, type))
{
  // Do nothing
}
//-----------------------------------------------------------------------------
petsc::Matrix::Matrix(Mat A, bool inc_ref_count) : _matA(A)
{
  if (!_matA)
    throw std::runtime_error("PETSc Mat must be initialised before wrapping");

  if (inc_ref_count)
  {
    common::petsc::check(PetscObjectReference((PetscObject)_matA),
                         "PetscObjectReference");
  }
}
//-----------------------------------------------------------------------------
petsc::Matrix::Matrix(Matrix&& A) noexcept
    : _matA(std::exchange(A._matA, nullptr))
{
}
//-----------------------------------------------------------------------------
petsc::Matrix::~Matrix()
{
  // Decrease reference count (PETSc will destroy object once reference
  // counts reached zero). Destructor is implicitly noexcept, so a
  // thrown error here calls std::terminate rather than propagating.
  if (_matA)
    common::petsc::check(MatDestroy(&_matA), "MatDestroy");
}
//-----------------------------------------------------------------------------
petsc::Matrix& petsc::Matrix::operator=(Matrix&& A) noexcept
{
  std::swap(_matA, A._matA);
  return *this;
}
//-----------------------------------------------------------------------------
std::array<std::int64_t, 2> petsc::Matrix::size() const
{
  assert(_matA);
  PetscInt m(0), n(0);
  common::petsc::check(MatGetSize(_matA, &m, &n), "MatGetSize");
  return {{m, n}};
}
//-----------------------------------------------------------------------------
Vec petsc::Matrix::create_vector(std::size_t dim) const
{
  assert(_matA);

  Vec x = nullptr;
  if (dim == 0)
    common::petsc::check(MatCreateVecs(_matA, nullptr, &x), "MatCreateVecs");
  else if (dim == 1)
    common::petsc::check(MatCreateVecs(_matA, &x, nullptr), "MatCreateVecs");
  else
  {
    spdlog::error("Cannot initialize PETSc vector to match PETSc matrix. "
                  "Dimension must be 0 or 1, not {}",
                  dim);
    throw std::runtime_error("Invalid dimension");
  }

  return x;
}
//-----------------------------------------------------------------------------
Mat petsc::Matrix::mat() const { return _matA; }
//-----------------------------------------------------------------------------
void petsc::Matrix::set_options_prefix(std::string_view options_prefix)
{
  assert(_matA);
  common::petsc::check(
      MatSetOptionsPrefix(_matA, std::string(options_prefix).c_str()),
      "MatSetOptionsPrefix");
}
//-----------------------------------------------------------------------------
std::string petsc::Matrix::get_options_prefix() const
{
  assert(_matA);
  const char* prefix = nullptr;
  common::petsc::check(MatGetOptionsPrefix(_matA, &prefix),
                       "MatGetOptionsPrefix");
  return prefix ? std::string(prefix) : std::string();
}
//-----------------------------------------------------------------------------
void petsc::Matrix::set_from_options()
{
  assert(_matA);
  common::petsc::check(MatSetFromOptions(_matA), "MatSetFromOptions");
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
petsc::KrylovSolver::KrylovSolver(MPI_Comm comm) : _ksp(nullptr)
{
  // Create PETSc KSP object
  common::petsc::check(KSPCreate(comm, &_ksp), "KSPCreate");
}
//-----------------------------------------------------------------------------
petsc::KrylovSolver::KrylovSolver(KSP ksp, bool inc_ref_count) : _ksp(ksp)
{
  if (!_ksp)
    throw std::runtime_error("PETSc KSP must be initialised before wrapping");

  if (inc_ref_count)
  {
    common::petsc::check(PetscObjectReference((PetscObject)_ksp),
                         "PetscObjectReference");
  }
}
//-----------------------------------------------------------------------------
petsc::KrylovSolver::KrylovSolver(KrylovSolver&& solver) noexcept
    : _ksp(std::exchange(solver._ksp, nullptr))
{
  // Do nothing
}
//-----------------------------------------------------------------------------
petsc::KrylovSolver::~KrylovSolver()
{
  // Destructor is implicitly noexcept, so a thrown error here calls
  // std::terminate rather than propagating
  if (_ksp)
    common::petsc::check(KSPDestroy(&_ksp), "KSPDestroy");
}
//-----------------------------------------------------------------------------
petsc::KrylovSolver&
petsc::KrylovSolver::operator=(KrylovSolver&& solver) noexcept
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
  common::petsc::check(KSPSetOperators(_ksp, A, P), "KSPSetOperators");
}
//-----------------------------------------------------------------------------
PetscInt petsc::KrylovSolver::solve(Vec x, const Vec b, bool transpose) const
{
  common::Timer timer("PETSc Krylov solver");
  assert(_ksp);
  assert(x);
  assert(b);

  // Solve linear system. With no operator set, PCGetOperators creates an
  // untyped Mat and setup fails on it, so KSPSolve errors rather than
  // silently solving
  spdlog::info("PETSc Krylov solver starting to solve system.");
  if (!transpose)
    common::petsc::check(KSPSolve(_ksp, b, x), "KSPSolve");
  else
    common::petsc::check(KSPSolveTranspose(_ksp, b, x), "KSPSolveTranspose");

  // Get the number of iterations
  PetscInt num_iterations = 0;
  common::petsc::check(KSPGetIterationNumber(_ksp, &num_iterations),
                       "KSPGetIterationNumber");

  // Check if the solution converged and warn if not. Note: this does
  // not throw on non-convergence -- the caller is responsible for
  // checking the convergence reason (via ksp()) if this matters for
  // its use case.
  KSPConvergedReason reason;
  common::petsc::check(KSPGetConvergedReason(_ksp, &reason),
                       "KSPGetConvergedReason");
  if (reason < 0)
  {
    const char* reason_str;
    common::petsc::check(KSPGetConvergedReasonString(_ksp, &reason_str),
                         "KSPGetConvergedReasonString");
    spdlog::warn("PETSc Krylov solver did not converge in {} iterations "
                 "(PETSc reason: {}).",
                 num_iterations, reason_str);
  }

  return num_iterations;
}
//-----------------------------------------------------------------------------
void petsc::KrylovSolver::set_options_prefix(std::string_view options_prefix)
{
  // Set options prefix
  assert(_ksp);
  common::petsc::check(
      KSPSetOptionsPrefix(_ksp, std::string(options_prefix).c_str()),
      "KSPSetOptionsPrefix");
}
//-----------------------------------------------------------------------------
std::string petsc::KrylovSolver::get_options_prefix() const
{
  assert(_ksp);
  const char* prefix = nullptr;
  common::petsc::check(KSPGetOptionsPrefix(_ksp, &prefix),
                       "KSPGetOptionsPrefix");
  return prefix ? std::string(prefix) : std::string();
}
//-----------------------------------------------------------------------------
void petsc::KrylovSolver::set_from_options() const
{
  assert(_ksp);
  common::petsc::check(KSPSetFromOptions(_ksp), "KSPSetFromOptions");
}
//-----------------------------------------------------------------------------
KSP petsc::KrylovSolver::ksp() const { return _ksp; }
//-----------------------------------------------------------------------------

#endif
