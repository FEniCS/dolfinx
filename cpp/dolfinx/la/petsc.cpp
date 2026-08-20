// Copyright (C) 2004-2018 Johan Hoffman, Johan Jansson, Anders Logg and
// Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#ifdef HAS_PETSC

#include "petsc.h"
#include "SparsityPattern.h"
#include "Vector.h"
#include "utils.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/common/log.h>
#include <format>
#include <numeric>
#include <ranges>
#include <stdexcept>
#include <utility>

using namespace dolfinx;
using namespace dolfinx::la;

//-----------------------------------------------------------------------------
#define CHECK_ERROR(NAME)                                                      \
  do                                                                           \
  {                                                                            \
    if (ierr != 0)                                                             \
      petsc::error(ierr, __FILE__, NAME);                                      \
  } while (0)

//-----------------------------------------------------------------------------
void la::petsc::error(PetscErrorCode error_code, std::string_view filename,
                      std::string_view petsc_function)
{
  // Fetch PETSc error description
  const char* desc;
  PetscErrorMessage(error_code, &desc, nullptr);

  // Log detailed error info
  spdlog::error("PETSc error in '{}', '{}'", filename, petsc_function);
  spdlog::error("PETSc error code '{}' '{}'", static_cast<int>(error_code),
                desc);
  throw std::runtime_error(
      std::format("Failed to successfully call PETSc function '{}'. PETSc "
                  "error code is: {}, {}",
                  petsc_function, static_cast<int>(error_code), desc));
}
//-----------------------------------------------------------------------------
std::vector<Vec>
la::petsc::create_vectors(MPI_Comm comm,
                          const std::vector<std::span<const PetscScalar>>& x)
{
  std::vector<Vec> v(x.size());
  for (std::size_t i = 0; i < v.size(); ++i)
  {
    PetscErrorCode ierr;
    ierr = VecCreateMPI(comm, x[i].size(), PETSC_DETERMINE, &v[i]);
    CHECK_ERROR("VecCreateMPI");
    PetscScalar* data;
    ierr = VecGetArray(v[i], &data);
    CHECK_ERROR("VecGetArray");
    std::ranges::copy(x[i], data);
    ierr = VecRestoreArray(v[i], &data);
    CHECK_ERROR("VecRestoreArray");
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
  PetscErrorCode ierr;

  // Get local size
  assert(range[1] >= range[0]);
  std::int32_t local_size = range[1] - range[0];

  Vec x = nullptr;
  std::vector<PetscInt> _ghosts(ghosts.begin(), ghosts.end());
  if (bs == 1)
  {
    ierr = VecCreateGhost(comm, local_size, PETSC_DETERMINE, _ghosts.size(),
                          _ghosts.data(), &x);
    CHECK_ERROR("VecCreateGhost");
  }
  else
  {
    ierr = VecCreateGhostBlock(comm, bs, bs * local_size, PETSC_DETERMINE,
                               _ghosts.size(), _ghosts.data(), &x);
    CHECK_ERROR("VecCreateGhostBlock");
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
  PetscErrorCode ierr;
  if (bs == 1)
  {
    ierr
        = VecCreateGhostWithArray(map.comm(), size_local, size_global,
                                  ghosts.size(), ghosts.data(), x.data(), &vec);
    CHECK_ERROR("VecCreateGhostWithArray");
  }
  else
  {
    ierr = VecCreateGhostBlockWithArray(map.comm(), bs, size_local, size_global,
                                        ghosts.size(), ghosts.data(), x.data(),
                                        &vec);
    CHECK_ERROR("VecCreateGhostBlockWithArray");
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
    PetscErrorCode ierr
        = ISCreateStride(PETSC_COMM_SELF, bs * size, offset, 1, &_is);
    CHECK_ERROR("ISCreateStride");
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
  PetscErrorCode ierr;
  Vec x_local;
  ierr = VecGhostGetLocalForm(x, &x_local);
  CHECK_ERROR("VecGhostGetLocalForm");
  PetscInt n = 0;
  ierr = VecGetSize(x_local, &n);
  CHECK_ERROR("VecGetSize");
  const PetscScalar* array = nullptr;
  ierr = VecGetArrayRead(x_local, &array);
  CHECK_ERROR("VecGetArrayRead");
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

  ierr = VecRestoreArrayRead(x_local, &array);
  CHECK_ERROR("VecRestoreArrayRead");
  ierr = VecGhostRestoreLocalForm(x, &x_local);
  CHECK_ERROR("VecGhostRestoreLocalForm");

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

  PetscErrorCode ierr;
  Vec x_local;
  ierr = VecGhostGetLocalForm(x, &x_local);
  CHECK_ERROR("VecGhostGetLocalForm");
  PetscInt n = 0;
  ierr = VecGetSize(x_local, &n);
  CHECK_ERROR("VecGetSize");
  PetscScalar* array = nullptr;
  ierr = VecGetArray(x_local, &array);
  CHECK_ERROR("VecGetArray");
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

  ierr = VecRestoreArray(x_local, &array);
  CHECK_ERROR("VecRestoreArray");
  ierr = VecGhostRestoreLocalForm(x, &x_local);
  CHECK_ERROR("VecGhostRestoreLocalForm");
}
//-----------------------------------------------------------------------------
Mat la::petsc::create_matrix(MPI_Comm comm, const SparsityPattern& sp,
                             std::optional<std::string_view> type)
{
  PetscErrorCode ierr;
  Mat A;
  ierr = MatCreate(comm, &A);
  if (ierr != 0)
    petsc::error(ierr, __FILE__, "MatCreate");

  // Get IndexMaps from sparsity pattern, and block size
  std::array maps = {sp.index_map(0), sp.index_map(1)};
  const std::array bs = {sp.block_size(0), sp.block_size(1)};

  if (type and !type->empty())
  {
    ierr = MatSetType(A, std::string(*type).c_str());
    if (ierr != 0)
      petsc::error(ierr, __FILE__, "MatSetType");
  }

  // Get global and local dimensions
  const std::int64_t M = bs[0] * maps[0]->size_global();
  const std::int64_t N = bs[1] * maps[1]->size_global();
  const std::int32_t m = bs[0] * maps[0]->size_local();
  const std::int32_t n = bs[1] * maps[1]->size_local();

  // Set matrix size
  ierr = MatSetSizes(A, m, n, M, N);
  if (ierr != 0)
    petsc::error(ierr, __FILE__, "MatSetSizes");

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
  ierr = MatXAIJSetPreallocation(A, _bs, _nnz_diag.data(), _nnz_offdiag.data(),
                                 nullptr, nullptr);
  if (ierr != 0)
    petsc::error(ierr, __FILE__, "MatXAIJSetPreallocation");

  // Set block sizes
  ierr = MatSetBlockSizes(A, bs[0], bs[1]);
  if (ierr != 0)
    petsc::error(ierr, __FILE__, "MatSetBlockSizes");

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
    std::vector<PetscInt> _map1 = build_l2g(*maps[1]);
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
                                         std::span<const Vec> basis)
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
void petsc::options::set(std::string option)
{
  petsc::options::set<std::string>(std::move(option), "");
}
//-----------------------------------------------------------------------------
void petsc::options::clear(std::string option)
{
  if (option[0] != '-')
    option = '-' + option;

  PetscErrorCode ierr;
  ierr = PetscOptionsClearValue(nullptr, option.c_str());
  if (ierr != 0)
    petsc::error(ierr, __FILE__, "PetscOptionsClearValue");
}
//-----------------------------------------------------------------------------
void petsc::options::clear()
{
  PetscErrorCode ierr = PetscOptionsClear(nullptr);
  if (ierr != 0)
    petsc::error(ierr, __FILE__, "PetscOptionsClear");
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
  assert(x);
  if (inc_ref_count)
    PetscObjectReference(reinterpret_cast<PetscObject>(_x));
}
//-----------------------------------------------------------------------------
petsc::Vector::Vector(Vector&& v) noexcept : _x(std::exchange(v._x, nullptr)) {}
//-----------------------------------------------------------------------------
petsc::Vector::~Vector()
{
  if (_x)
    VecDestroy(&_x);
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
  PetscErrorCode ierr = VecDuplicate(_x, &_y);
  CHECK_ERROR("VecDuplicate");
  ierr = VecCopy(_x, _y);
  CHECK_ERROR("VecCopy");
  return Vector(_y, false);
}
//-----------------------------------------------------------------------------
std::int64_t petsc::Vector::size() const
{
  assert(_x);
  PetscInt n = 0;
  PetscErrorCode ierr = VecGetSize(_x, &n);
  CHECK_ERROR("VecGetSize");
  return n;
}
//-----------------------------------------------------------------------------
std::int32_t petsc::Vector::local_size() const
{
  assert(_x);
  PetscInt n = 0;
  PetscErrorCode ierr = VecGetLocalSize(_x, &n);
  CHECK_ERROR("VecGetLocalSize");
  return n;
}
//-----------------------------------------------------------------------------
std::array<std::int64_t, 2> petsc::Vector::local_range() const
{
  assert(_x);
  PetscInt n0, n1;
  PetscErrorCode ierr = VecGetOwnershipRange(_x, &n0, &n1);
  CHECK_ERROR("VecGetOwnershipRange");
  assert(n0 <= n1);
  return {n0, n1};
}
//-----------------------------------------------------------------------------
MPI_Comm petsc::Vector::comm() const
{
  assert(_x);
  MPI_Comm mpi_comm = MPI_COMM_NULL;
  PetscErrorCode ierr
      = PetscObjectGetComm(reinterpret_cast<PetscObject>(_x), &mpi_comm);
  CHECK_ERROR("PetscObjectGetComm");
  return mpi_comm;
}
//-----------------------------------------------------------------------------
void petsc::Vector::set_options_prefix(std::string_view options_prefix)
{
  assert(_x);
  PetscErrorCode ierr
      = VecSetOptionsPrefix(_x, std::string(options_prefix).c_str());
  CHECK_ERROR("VecSetOptionsPrefix");
}
//-----------------------------------------------------------------------------
std::string petsc::Vector::get_options_prefix() const
{
  assert(_x);
  const char* prefix = nullptr;
  PetscErrorCode ierr = VecGetOptionsPrefix(_x, &prefix);
  CHECK_ERROR("VecGetOptionsPrefix");
  // PETSc reports an unset prefix as a null pointer
  return prefix ? std::string(prefix) : std::string();
}
//-----------------------------------------------------------------------------
void petsc::Vector::set_from_options()
{
  assert(_x);
  PetscErrorCode ierr = VecSetFromOptions(_x);
  CHECK_ERROR("VecSetFromOptions");
}
//-----------------------------------------------------------------------------
Vec petsc::Vector::vec() const { return _x; }
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
petsc::Operator::Operator(Mat A, bool inc_ref_count) : _matA(A)
{
  assert(A);
  if (inc_ref_count)
    PetscObjectReference(reinterpret_cast<PetscObject>(_matA));
}
//-----------------------------------------------------------------------------
petsc::Operator::Operator(Operator&& A) noexcept
    : _matA(std::exchange(A._matA, nullptr))
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
petsc::Operator& petsc::Operator::operator=(Operator&& A) noexcept
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
    petsc::error(ierr, __FILE__, "MatGetSize");
  return {{m, n}};
}
//-----------------------------------------------------------------------------
Vec petsc::Operator::create_vector(std::size_t dim) const
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
    spdlog::error("Cannot initialize PETSc vector to match PETSc matrix. "
                  "Dimension must be 0 or 1, not {}",
                  dim);
    throw std::runtime_error("Invalid dimension");
  }

  return x;
}
//-----------------------------------------------------------------------------
Mat petsc::Operator::mat() const { return _matA; }
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
petsc::Matrix::Matrix(MPI_Comm comm, const SparsityPattern& sp,
                      std::optional<std::string_view> type)
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
  PetscReal value = 0;
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
void petsc::Matrix::set_options_prefix(std::string_view options_prefix)
{
  assert(_matA);
  MatSetOptionsPrefix(_matA, std::string(options_prefix).c_str());
}
//-----------------------------------------------------------------------------
std::string petsc::Matrix::get_options_prefix() const
{
  assert(_matA);
  const char* prefix = nullptr;
  MatGetOptionsPrefix(_matA, &prefix);
  // PETSc reports an unset prefix as a null pointer
  return prefix ? std::string(prefix) : std::string();
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
  // Create PETSc KSP object
  PetscErrorCode ierr = KSPCreate(comm, &_ksp);
  if (ierr != 0)
    petsc::error(ierr, __FILE__, "KSPCreate");
}
//-----------------------------------------------------------------------------
petsc::KrylovSolver::KrylovSolver(KSP ksp, bool inc_ref_count) : _ksp(ksp)
{
  assert(_ksp);
  if (inc_ref_count)
  {
    PetscErrorCode ierr
        = PetscObjectReference(reinterpret_cast<PetscObject>(_ksp));
    if (ierr != 0)
      petsc::error(ierr, __FILE__, "PetscObjectReference");
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
  if (_ksp)
    KSPDestroy(&_ksp);
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
  PetscErrorCode ierr = KSPSetOperators(_ksp, A, P);
  if (ierr != 0)
    petsc::error(ierr, __FILE__, "KSPSetOperators");
}
//-----------------------------------------------------------------------------
int petsc::KrylovSolver::solve(Vec x, const Vec b, bool transpose)
{
  common::Timer timer("PETSc Krylov solver");
  assert(x);
  assert(b);

  // Get PETSc operators
  Mat _A, _P;
  KSPGetOperators(_ksp, &_A, &_P);
  assert(_A);

  PetscErrorCode ierr;

  // Solve linear system
  spdlog::info("PETSc Krylov solver starting to solve system.");

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

  // Get the number of iterations
  PetscInt num_iterations = 0;
  ierr = KSPGetIterationNumber(_ksp, &num_iterations);
  if (ierr != 0)
    petsc::error(ierr, __FILE__, "KSPGetIterationNumber");

  // Check if the solution converged and warn if not. Note: this does
  // not throw on non-convergence -- the caller is responsible for
  // checking the convergence reason (via ksp()) if this matters for
  // its use case.
  KSPConvergedReason reason;
  ierr = KSPGetConvergedReason(_ksp, &reason);
  if (ierr != 0)
    petsc::error(ierr, __FILE__, "KSPGetConvergedReason");
  if (reason < 0)
  {
    const char* reason_str;
    ierr = KSPGetConvergedReasonString(_ksp, &reason_str);
    if (ierr != 0)
      petsc::error(ierr, __FILE__, "KSPGetConvergedReasonString");
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
  PetscErrorCode ierr
      = KSPSetOptionsPrefix(_ksp, std::string(options_prefix).c_str());
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
  // PETSc reports an unset prefix as a null pointer
  return prefix ? std::string(prefix) : std::string();
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

#endif
