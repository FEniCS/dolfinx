// Copyright (C) 2004-2026 Johan Hoffman, Johan Jansson, Anders Logg,
// Garth N. Wells and Jack S. Hale
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#ifdef HAS_PETSC

#include "Vector.h"
#include <array>
#include <cassert>
#include <cstdint>
#include <dolfinx/common/petsc.h>
#include <functional>
#include <optional>
#include <petscksp.h>
#include <petscmat.h>
#include <petscvec.h>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace dolfinx::common
{
class IndexMap;
} // namespace dolfinx::common

namespace dolfinx::la
{
class SparsityPattern;

/// @brief PETSc linear algebra functions
namespace petsc
{
/// @brief Create PETSc vectors from the local data. The data is
/// copied into the PETSc vectors and is not shared. Each vector's
/// global size is determined by summing the corresponding local size
/// across all ranks in `comm`.
/// @note Caller is responsible for destroying the returned object
/// @param[in] comm The MPI communicator
/// @param[in] x The vector data owned by the calling rank
/// @return Array of PETSc vectors
std::vector<Vec>
create_vectors(MPI_Comm comm,
               const std::vector<std::span<const PetscScalar>>& x);

/// @brief Create a ghosted PETSc Vec.
/// @note Caller is responsible for destroying the returned object
/// @param[in] map The index map describing the parallel layout (by block)
/// @param[in] bs The block size
/// @returns A PETSc Vec
Vec create_vector(const common::IndexMap& map, int bs);

/// @brief Create a ghosted PETSc Vec from a local range and ghost
/// indices.
/// @note Caller is responsible for freeing the returned object
/// @param[in] comm The MPI communicator
/// @param[in] range The local ownership range (by blocks)
/// @param[in] ghosts Ghost blocks
/// @param[in] bs The block size. The total number of local entries is
/// `bs * (range[1] - range[0])`.
/// @returns A PETSc Vec
Vec create_vector(MPI_Comm comm, std::array<std::int64_t, 2> range,
                  std::span<const std::int64_t> ghosts, int bs);

/// @brief Create a PETSc Vec that wraps the data in an array.
/// @param[in] map The index map that describes the parallel layout of
/// the distributed vector (by block)
/// @param[in] bs Block size
/// @param[in] x The local part of the vector, including ghost entries.
/// Must have size at least `bs * (map.size_local() + map.num_ghosts())`.
/// @return A PETSc Vec object that shares the data in @p x
/// @note The array `x` must be kept alive to use the PETSc Vec object
/// @note The caller should call VecDestroy to free the return PETSc
/// vector
Vec create_vector_wrap(const common::IndexMap& map, int bs,
                       std::span<const PetscScalar> x);

/// @brief Create a PETSc Vec that wraps the data in an array.
/// @param[in] x The vector to be wrapped
/// @return A PETSc Vec object that shares the data in @p x
template <class V>
Vec create_vector_wrap(const la::Vector<V>& x)
{
  assert(x.index_map());
  return create_vector_wrap(*x.index_map(), x.bs(), x.array());
}

/// @brief Compute PETSc IndexSets (IS) for a stack of index maps.
///
/// If `map[0] = {0, 1, 2, 3, 4, 5, 6}` and `map[1] = {0, 1, 2, 4}` (in
/// local indices) then `IS[0] = {0, 1, 2, 3, 4, 5, 6}` and
/// `IS[1] = {7, 8, 9, 10}`.
///
/// @todo This function could take just the local sizes.
///
/// @note The caller is responsible for destruction of each IS.
///
/// @param[in] maps Vector of IndexMaps and corresponding block sizes
/// @return Vector of PETSc Index Sets, created on` PETSC_COMM_SELF`
std::vector<IS> create_index_sets(
    const std::vector<
        std::pair<std::reference_wrapper<const common::IndexMap>, int>>& maps);

/// Copy blocks from Vec into local arrays
std::vector<std::vector<PetscScalar>> get_local_vectors(
    const Vec x,
    const std::vector<
        std::pair<std::reference_wrapper<const common::IndexMap>, int>>& maps);

/// Scatter local vectors to Vec
void scatter_local_vectors(
    Vec x, const std::vector<std::span<const PetscScalar>>& x_b,
    const std::vector<
        std::pair<std::reference_wrapper<const common::IndexMap>, int>>& maps);

/// @brief Create a PETSc Mat. Caller is responsible for destroying the
/// returned object.
///
/// The returned matrix is *preallocated*, not populated: room is
/// reserved for each row's non-zeros, but an entry does not exist until
/// a value is inserted into it.
///
/// @note For matrix types that support `MAT_IGNORE_ZERO_ENTRIES`
/// (`MATAIJ`, `MATSELL`, and `MATIS`), repeated re-assembly with a fixed
/// insertion pattern can be substantially faster when the option is
/// enabled. It prevents most exact-zero additions from creating or
/// searching for an entry. A zero on the locally owned diagonal is
/// retained. Adding zero cannot change a matrix, so the assembled result
/// is unchanged.
///
/// How much this saves is a property of the problem and worth measuring
/// first, because the zeros have two quite different origins. Those
/// coming from the form's block structure are reliable: a vector
/// Laplacian does not couple components, so `1 - 1/d` of each element
/// block vanishes for `d` components -- measured 67% for `d = 3`, and
/// 37% for a Taylor-Hood Stokes system, on meshes with no special
/// geometry. Those coming from the geometry are not: a scalar Laplacian
/// has 34% zero entries on a structured simplex mesh, where each cell
/// has orthogonal edge pairs, but 0.1% once the vertices are perturbed.
/// A non-linear Jacobian couples every local degree of freedom to every
/// other and typically has none.
///
/// When to enable it depends on whether the matrix is assembled more
/// than once, because the option also decides whether an entry is ever
/// *created*.
///
/// For a matrix that is re-assembled, enable it only *after* a full
/// assembly has completed:
/// @code
///   Mat A = la::petsc::create_matrix(comm, sp);
///   // ... assemble into A, ending in MAT_FINAL_ASSEMBLY ...
///   MatSetOption(A, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE);
///   // subsequent MatZeroEntries + re-assembly skip zero insertions
/// @endcode
/// The first assembly creates every location that it inserts, including
/// those receiving only zero values, and `MatAssemblyEnd` does not
/// discard stored zeros. `MatZeroEntries` always preserves the sparse
/// pattern; this is independent of `MAT_KEEP_NONZERO_PATTERN`, which is
/// set here for `MatZeroRows`.
///
/// Enabling it before the first assembly instead means most
/// off-diagonal entries that are zero then are never created, leaving
/// the matrix with a smaller non-zero pattern than `sp` describes. That
/// is a bug for a matrix that is re-assembled -- nothing appears wrong,
/// because the missing entries are zero, until a later assembly produces
/// a non-zero at one of them and `MAT_NEW_NONZERO_ALLOCATION_ERR` (set
/// here) raises an error.
///
/// For a matrix that is assembled once and then solved with, it is
/// instead a deliberate optimisation: the dropped entries are never
/// needed, and the result is a smaller operator to store and to apply.
/// `demo_stokes.py` does this, dropping 42% of the entries of a
/// Taylor-Hood operator.
///
/// @param[in] comm The MPI communicator
/// @param[in] sp The sparsity pattern that determines the layout and
/// non-zero structure of the matrix
/// @param[in] type The PETSc Mat type to create. If `std::nullopt` or
/// an empty string, the PETSc default is used.
Mat create_matrix(MPI_Comm comm, const SparsityPattern& sp,
                  std::optional<std::string_view> type = std::nullopt);

/// @brief Create PETSc MatNullSpace. Caller is responsible for
/// destruction returned object.
/// @param[in] comm The MPI communicator
/// @param[in] basis The nullspace basis vectors
/// @return A PETSc nullspace object
MatNullSpace create_nullspace(MPI_Comm comm, std::span<const Vec> basis);

/// A simple wrapper for a PETSc vector pointer (Vec). Its main purpose
/// is to assist with memory/lifetime management of PETSc Vec objects.
///
/// Access the underlying PETSc Vec pointer using the function
/// Vector::vec() and use the full PETSc interface.
class Vector
{
public:
  /// @brief Create a vector.
  /// @note Collective
  /// @param[in] map Index map describing the parallel layout
  /// @param[in] bs the block size
  Vector(const common::IndexMap& map, int bs);

  // Delete copy constructor to avoid accidental copying of 'heavy' data
  Vector(const Vector& x) = delete;

  /// Move constructor
  Vector(Vector&& x) noexcept;

  /// @brief Create holder of a PETSc Vec object/pointer. The Vec x
  /// object should already be created. If inc_ref_count is true, the
  /// reference counter of the Vec object will be increased. The Vec
  /// reference count will always be decreased upon destruction of the
  /// PETScVector.
  ///
  /// @note Collective
  ///
  /// @param[in] x The PETSc Vec
  /// @param[in] inc_ref_count True if the reference count of `x` should
  /// be incremented
  Vector(Vec x, bool inc_ref_count);

  /// Destructor
  ~Vector();

  // Assignment operator (disabled)
  Vector& operator=(const Vector& x) = delete;

  /// Move Assignment operator
  Vector& operator=(Vector&& x) noexcept;

  /// @brief Create a copy of the vector.
  /// @note Collective
  Vector copy() const;

  /// Return global size of the vector
  std::int64_t size() const;

  /// Return local size of vector (belonging to the call rank)
  std::int32_t local_size() const;

  /// Return ownership range for calling rank
  std::array<std::int64_t, 2> local_range() const;

  /// Return MPI communicator
  MPI_Comm comm() const;

  /// Sets the prefix used by PETSc when searching the options database
  void set_options_prefix(std::string_view options_prefix);

  /// Returns the prefix used by PETSc when searching the options
  /// database
  std::string get_options_prefix() const;

  /// Call PETSc function VecSetFromOptions on the underlying Vec object
  void set_from_options();

  /// Return pointer to PETSc Vec object
  Vec vec() const;

private:
  // PETSc Vec pointer
  Vec _x;
};

/// It is a simple wrapper for a PETSc matrix pointer (Mat). Its main
/// purpose is to assist memory management of PETSc Mat objects.
///
/// For advanced usage, access the PETSc Mat pointer using the function
/// mat() and use the standard PETSc interface.
class Matrix
{
public:
  /// @brief Return a function with an interface for adding or inserting
  /// values into the matrix A (calls MatSetValuesLocal).
  ///
  /// @param[in] A The matrix to set values in
  /// @param[in] mode The PETSc insert mode (ADD_VALUES, INSERT_VALUES, ...)
  static auto set_fn(Mat A, InsertMode mode)
  {
    return [A, mode, cache = std::vector<PetscInt>()](
               std::span<const std::int32_t> rows,
               std::span<const std::int32_t> cols,
               std::span<const PetscScalar> vals) mutable -> int
    {
      PetscErrorCode ierr;
#ifdef PETSC_USE_64BIT_INDICES
      cache.resize(rows.size() + cols.size());
      std::ranges::copy(rows, cache.begin());
      std::ranges::copy(cols, std::next(cache.begin(), rows.size()));
      const PetscInt* _rows = cache.data();
      const PetscInt* _cols = cache.data() + rows.size();
      ierr = MatSetValuesLocal(A, rows.size(), _rows, cols.size(), _cols,
                               vals.data(), mode);
#else
      ierr = MatSetValuesLocal(A, rows.size(), rows.data(), cols.size(),
                               cols.data(), vals.data(), mode);
#endif

#ifndef NDEBUG
      common::petsc::check(ierr, "MatSetValuesLocal");
#endif
      return ierr;
    };
  }

  /// @brief Return a function with an interface for adding or
  /// inserting values into the matrix A using blocked indices (calls
  /// MatSetValuesBlockedLocal).
  /// @param[in] A The matrix to set values in
  /// @param[in] mode The PETSc insert mode (ADD_VALUES, INSERT_VALUES, ...)
  static auto set_block_fn(Mat A, InsertMode mode)
  {
    return [A, mode, cache = std::vector<PetscInt>()](
               std::span<const std::int32_t> rows,
               std::span<const std::int32_t> cols,
               std::span<const PetscScalar> vals) mutable -> int
    {
      PetscErrorCode ierr;
#ifdef PETSC_USE_64BIT_INDICES
      cache.resize(rows.size() + cols.size());
      std::ranges::copy(rows, cache.begin());
      std::ranges::copy(cols, std::next(cache.begin(), rows.size()));
      const PetscInt* _rows = cache.data();
      const PetscInt* _cols = cache.data() + rows.size();
      ierr = MatSetValuesBlockedLocal(A, rows.size(), _rows, cols.size(), _cols,
                                      vals.data(), mode);
#else
      ierr = MatSetValuesBlockedLocal(A, rows.size(), rows.data(), cols.size(),
                                      cols.data(), vals.data(), mode);
#endif

#ifndef NDEBUG
      common::petsc::check(ierr, "MatSetValuesBlockedLocal");
#endif
      return ierr;
    };
  }

  /// @brief Return a function with an interface for adding or inserting
  /// blocked values to the matrix A using non-blocked insertion (calls
  /// MatSetValuesLocal).
  ///
  /// Internally it expands the blocked indices into non-blocked arrays.
  ///
  /// @param[in] A The matrix to set values in
  /// @param[in] bs0 Block size for the matrix rows
  /// @param[in] bs1 Block size for the matrix columns
  /// @param[in] mode The PETSc insert mode (ADD_VALUES, INSERT_VALUES, ...)
  static auto set_block_expand_fn(Mat A, int bs0, int bs1, InsertMode mode)
  {
    return [A, bs0, bs1, mode, cache0 = std::vector<PetscInt>(),
            cache1 = std::vector<PetscInt>()](
               std::span<const std::int32_t> rows,
               std::span<const std::int32_t> cols,
               std::span<const PetscScalar> vals) mutable -> int
    {
      PetscErrorCode ierr;
      cache0.resize(bs0 * rows.size());
      cache1.resize(bs1 * cols.size());
      for (std::size_t i = 0; i < rows.size(); ++i)
        for (int k = 0; k < bs0; ++k)
          cache0[bs0 * i + k] = bs0 * rows[i] + k;

      for (std::size_t i = 0; i < cols.size(); ++i)
        for (int k = 0; k < bs1; ++k)
          cache1[bs1 * i + k] = bs1 * cols[i] + k;

      ierr = MatSetValuesLocal(A, cache0.size(), cache0.data(), cache1.size(),
                               cache1.data(), vals.data(), mode);
#ifndef NDEBUG
      common::petsc::check(ierr, "MatSetValuesLocal");
#endif
      return ierr;
    };
  }

  /// Create holder for a PETSc Mat object from a sparsity pattern
  Matrix(MPI_Comm comm, const SparsityPattern& sp,
         std::optional<std::string_view> type = std::nullopt);

  /// @brief Create holder of a PETSc Mat object/pointer. The Mat A object
  /// should already be created.
  /// @param[in] A PETSc Mat object, which must already have been
  /// created. The reference count of `A` is always decreased when this
  /// Matrix is destroyed.
  /// @param[in] inc_ref_count True if the reference count of `A` should
  /// be incremented.
  Matrix(Mat A, bool inc_ref_count);

  // Copy constructor (deleted)
  Matrix(const Matrix& A) = delete;

  /// Move constructor
  Matrix(Matrix&& A) noexcept;

  /// Destructor
  ~Matrix();

  // Assignment operator (deleted)
  Matrix& operator=(const Matrix& A) = delete;

  /// Move assignment operator
  Matrix& operator=(Matrix&& A) noexcept;

  /// Return number of rows and columns (num_rows, num_cols). PETSc
  /// returns -1 if size has not been set.
  std::array<std::int64_t, 2> size() const;

  /// @brief Initialize vector to be compatible with the matrix-vector
  /// product y = Ax. In the parallel case, size and layout are both
  /// important.
  ///
  /// @param[in] dim The dimension (axis): dim = 0 --> z = y, dim = 1
  /// --> z = x
  Vec create_vector(std::size_t dim) const;

  /// Return PETSc Mat pointer
  Mat mat() const;

  //--- Special PETSc Functions ---

  /// Sets the prefix used by PETSc when searching the options
  /// database
  void set_options_prefix(std::string_view options_prefix);

  /// Returns the prefix used by PETSc when searching the options
  /// database
  std::string get_options_prefix() const;

  /// Call PETSc function MatSetFromOptions on the PETSc Mat object
  void set_from_options();

private:
  // PETSc Mat pointer
  Mat _matA;
};

/// This class implements Krylov methods for linear systems of the form
/// Ax = b. It is a wrapper for the Krylov solvers of PETSc.
class KrylovSolver
{
public:
  /// @brief Create a Krylov solver.
  /// @param[in] comm MPI communicator.
  explicit KrylovSolver(MPI_Comm comm);

  /// @brief Create solver wrapper of a PETSc KSP object.
  /// @param[in] ksp PETSc KSP object, which must already have been
  /// created. The reference count of `ksp` is always decreased when this
  /// KrylovSolver is destroyed.
  /// @param[in] inc_ref_count True if the reference count of `ksp`
  /// should be incremented.
  KrylovSolver(KSP ksp, bool inc_ref_count);

  // Copy constructor (deleted)
  KrylovSolver(const KrylovSolver& solver) = delete;

  /// Move constructor
  KrylovSolver(KrylovSolver&& solver) noexcept;

  /// Destructor
  ~KrylovSolver();

  // Assignment operator (deleted)
  KrylovSolver& operator=(const KrylovSolver&) = delete;

  /// Move assignment
  KrylovSolver& operator=(KrylovSolver&& solver) noexcept;

  /// Set operator (Mat)
  void set_operator(const Mat A);

  /// Set operator and preconditioner matrix (Mat)
  void set_operators(const Mat A, const Mat P);

  /// @brief Solve linear system Ax = b (A^t x = b if `transpose` is
  /// true).
  ///
  /// Non-convergence is not treated as an error (a warning is
  /// logged); check the returned convergence reason.
  ///
  /// @return The PETSc convergence reason (positive on convergence,
  /// negative on divergence).
  [[nodiscard(
      "check the converged reason - positive on convergence, negative on "
      "divergence")]] KSPConvergedReason
  solve(Vec x, const Vec b, bool transpose = false);

  /// Sets the prefix used by PETSc when searching the PETSc options
  /// database
  void set_options_prefix(std::string_view options_prefix);

  /// Returns the prefix used by PETSc when searching the PETSc options
  /// database
  std::string get_options_prefix() const;

  /// Set options from PETSc options database
  void set_from_options() const;

  /// Return PETSc KSP pointer
  KSP ksp() const;

private:
  // PETSc solver pointer
  KSP _ksp;
};
} // namespace petsc
} // namespace dolfinx::la

#endif
