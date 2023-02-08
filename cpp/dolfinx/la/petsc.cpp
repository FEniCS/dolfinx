// Copyright (C) 2004-2018 Johan Hoffman, Johan Jansson, Anders Logg and
// Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "petsc.h"
#include "SparsityPattern.h"
#include "Vector.h"
#include "utils.h"
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/common/log.h>
#include <iostream>
#include <sstream>

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
void la::petsc::error(int error_code, std::string filename,
                      std::string petsc_function)
{
  // Fetch PETSc error description
  const char* desc;
  PetscErrorMessage(error_code, &desc, nullptr);

  // Log detailed error info
  DLOG(INFO) << "PETSc error in '" << filename.c_str() << "', '"
             << petsc_function.c_str() << "'";
  DLOG(INFO) << "PETSc error code '" << error_code << "' (" << desc << ".";
  throw std::runtime_error("Failed to successfully call PETSc function '"
                           + petsc_function + "'. PETSc error code is: "
                           + std ::to_string(error_code) + ", "
                           + std::string(desc));
}
//-----------------------------------------------------------------------------
std::vector<Vec>
la::petsc::create_vectors(MPI_Comm comm,
                          const std::vector<std::span<const PetscScalar>>& x)
{
  std::vector<Vec> v(x.size());
  for (std::size_t i = 0; i < v.size(); ++i)
  {
    VecCreateMPI(comm, x[i].size(), PETSC_DETERMINE, &v[i]);
    PetscScalar* data;
    VecGetArray(v[i], &data);
    std::copy(x[i].begin(), x[i].end(), data);
    VecRestoreArray(v[i], &data);
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
  const std::int32_t local_size = range[1] - range[0];

  Vec x;
  const std::vector<PetscInt> _ghosts(ghosts.begin(), ghosts.end());
  ierr = VecCreateGhostBlock(comm, bs, bs * local_size, PETSC_DETERMINE,
                             _ghosts.size(), _ghosts.data(), &x);
  CHECK_ERROR("VecCreateGhostBlock");
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
  Vec vec;
  VecCreateGhostBlockWithArray(map.comm(), bs, size_local, size_global,
                               ghosts.size(), ghosts.data(), x.data(), &vec);
  return vec;
}
//-----------------------------------------------------------------------------
std::vector<IS> la::petsc::create_index_sets(
    const std::vector<
        std::pair<std::reference_wrapper<const common::IndexMap>, int>>& maps)
{
  std::vector<IS> is;
  std::int64_t offset = 0;
  for (auto& map : maps)
  {
    const int bs = map.second;
    const std::int32_t size
        = map.first.get().size_local() + map.first.get().num_ghosts();
    IS _is;
    ISCreateStride(PETSC_COMM_SELF, bs * size, offset, 1, &_is);
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
  for (auto& map : maps)
    offset_owned += map.first.get().size_local() * map.second;

  // Unwrap PETSc vector
  Vec x_local;
  VecGhostGetLocalForm(x, &x_local);
  PetscInt n = 0;
  VecGetSize(x_local, &n);
  const PetscScalar* array = nullptr;
  VecGetArrayRead(x_local, &array);
  std::span _x(array, n);

  // Copy PETSc Vec data in to local vectors
  std::vector<std::vector<PetscScalar>> x_b;
  int offset = 0;
  int offset_ghost = offset_owned; // Ghost DoFs start after owned
  for (auto map : maps)
  {
    const std::int32_t size_owned = map.first.get().size_local() * map.second;
    const std::int32_t size_ghost = map.first.get().num_ghosts() * map.second;

    x_b.emplace_back(size_owned + size_ghost);
    std::copy_n(std::next(_x.begin(), offset), size_owned, x_b.back().begin());
    std::copy_n(std::next(_x.begin(), offset_ghost), size_ghost,
                std::next(x_b.back().begin(), size_owned));

    offset += size_owned;
    offset_ghost += size_ghost;
  }

  VecRestoreArrayRead(x_local, &array);
  VecGhostRestoreLocalForm(x, &x_local);

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
  for (auto& map : maps)
    offset_owned += map.first.get().size_local() * map.second;

  Vec x_local;
  VecGhostGetLocalForm(x, &x_local);
  PetscInt n = 0;
  VecGetSize(x_local, &n);
  PetscScalar* array = nullptr;
  VecGetArray(x_local, &array);
  std::span _x(array, n);

  // Copy local vectors into PETSc Vec
  int offset = 0;
  int offset_ghost = offset_owned; // Ghost DoFs start after owned
  for (std::size_t i = 0; i < maps.size(); ++i)
  {
    const std::int32_t size_owned
        = maps[i].first.get().size_local() * maps[i].second;
    const std::int32_t size_ghost
        = maps[i].first.get().num_ghosts() * maps[i].second;

    std::copy_n(x_b[i].begin(), size_owned, std::next(_x.begin(), offset));
    std::copy_n(std::next(x_b[i].begin(), size_owned), size_ghost,
                std::next(_x.begin(), offset_ghost));

    offset += size_owned;
    offset_ghost += size_ghost;
  }

  VecRestoreArray(x_local, &array);
  VecGhostRestoreLocalForm(x, &x_local);
}
//-----------------------------------------------------------------------------
Mat la::petsc::create_matrix(MPI_Comm comm, const SparsityPattern& sp,
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
      _nnz_diag[i] = sp.nnz_diag(i);
    for (std::size_t i = 0; i < _nnz_offdiag.size(); ++i)
      _nnz_offdiag[i] = sp.nnz_off_diag(i);
  }
  else
  {
    // Expand for block size 1
    _nnz_diag.resize(maps[0]->size_local() * bs[0]);
    _nnz_offdiag.resize(maps[0]->size_local() * bs[0]);
    for (std::size_t i = 0; i < _nnz_diag.size(); ++i)
      _nnz_diag[i] = bs[1] * sp.nnz_diag(i / bs[0]);
    for (std::size_t i = 0; i < _nnz_offdiag.size(); ++i)
      _nnz_offdiag[i] = bs[1] * sp.nnz_off_diag(i / bs[0]);
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
  petsc::options::set<std::string>(option, "");
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
    PetscObjectReference((PetscObject)_x);
}
//-----------------------------------------------------------------------------
petsc::Vector::Vector(Vector&& v) : _x(std::exchange(v._x, nullptr)) {}
//-----------------------------------------------------------------------------
petsc::Vector::~Vector()
{
  if (_x)
    VecDestroy(&_x);
}
//-----------------------------------------------------------------------------
petsc::Vector& petsc::Vector::operator=(Vector&& v)
{
  std::swap(_x, v._x);
  return *this;
}
//-----------------------------------------------------------------------------
petsc::Vector petsc::Vector::copy() const
{
  Vec _y;
  VecDuplicate(_x, &_y);
  VecCopy(_x, _y);
  Vector y(_y, true);
  VecDestroy(&_y);
  return y;
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
  PetscErrorCode ierr = PetscObjectGetComm((PetscObject)(_x), &mpi_comm);
  CHECK_ERROR("PetscObjectGetComm");
  return mpi_comm;
}
//-----------------------------------------------------------------------------
void petsc::Vector::set_options_prefix(std::string options_prefix)
{
  assert(_x);
  PetscErrorCode ierr = VecSetOptionsPrefix(_x, options_prefix.c_str());
  CHECK_ERROR("VecSetOptionsPrefix");
}
//-----------------------------------------------------------------------------
std::string petsc::Vector::get_options_prefix() const
{
  assert(_x);
  const char* prefix = nullptr;
  PetscErrorCode ierr = VecGetOptionsPrefix(_x, &prefix);
  CHECK_ERROR("VecGetOptionsPrefix");
  return std::string(prefix);
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
    LOG(ERROR) << "Cannot initialize PETSc vector to match PETSc matrix. "
               << "Dimension must be 0 or 1, not " << dim;
    throw std::runtime_error("Invalid dimension");
  }

  return x;
}
//-----------------------------------------------------------------------------
Mat petsc::Operator::mat() const { return _matA; }
//-----------------------------------------------------------------------------
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
  PetscErrorCode ierr = KSPSetOperators(_ksp, A, P);
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

  PetscErrorCode ierr;

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
