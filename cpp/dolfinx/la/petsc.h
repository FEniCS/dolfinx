// Copyright (C) 2004-2018 Johan Hoffman, Johan Jansson, Anders Logg and
// Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Vector.h"
#include "utils.h"
#include <boost/lexical_cast.hpp>
#include <functional>
#include <petscksp.h>
#include <petscmat.h>
#include <petscoptions.h>
#include <petscvec.h>
#include <span>
#include <string>
#include <vector>

namespace dolfinx::common
{
class IndexMap;
} // namespace dolfinx::common

namespace dolfinx::la
{
class SparsityPattern;

namespace petsc
{
/// Print error message for PETSc calls that return an error
void error(int error_code, std::string filename, std::string petsc_function);

/// Create PETsc vectors from the local data. The data is copied into
/// the PETSc vectors and is not shared.
/// @note Caller is responsible for destroying the returned object
/// @param[in] comm The MPI communicator
/// @param[in] x The vector data owned by the calling rank. All
/// components must have the same length.
/// @return Array of PETSc vectors
std::vector<Vec>
create_vectors(MPI_Comm comm,
               const std::vector<std::span<const PetscScalar>>& x);

/// Create a ghosted PETSc Vec
/// @note Caller is responsible for destroying the returned object
/// @param[in] map The index map describing the parallel layout (by block)
/// @param[in] bs The block size
/// @returns A PETSc Vec
Vec create_vector(const common::IndexMap& map, int bs);

/// Create a ghosted PETSc Vec from a local range and ghost indices
/// @note Caller is responsible for freeing the returned object
/// @param[in] comm The MPI communicator
/// @param[in] range The local ownership range (by blocks)
/// @param[in] ghosts Ghost blocks
/// @param[in] bs The block size. The total number of local entries is
/// `bs * (range[1] - range[0])`.
/// @returns A PETSc Vec
Vec create_vector(MPI_Comm comm, std::array<std::int64_t, 2> range,
                  std::span<const std::int64_t> ghosts, int bs);

/// Create a PETSc Vec that wraps the data in an array
/// @param[in] map The index map that describes the parallel layout of
/// the distributed vector (by block)
/// @param[in] bs Block size
/// @param[in] x The local part of the vector, including ghost entries
/// @return A PETSc Vec object that shares the data in @p x
/// @note The array `x` must be kept alive to use the PETSc Vec object
/// @note The caller should call VecDestroy to free the return PETSc
/// vector
Vec create_vector_wrap(const common::IndexMap& map, int bs,
                       std::span<const PetscScalar> x);

/// Create a PETSc Vec that wraps the data in an array
/// @param[in] x The vector to be wrapped
/// @return A PETSc Vec object that shares the data in @p x
template <typename Allocator>
Vec create_vector_wrap(const la::Vector<PetscScalar, Allocator>& x)
{
  assert(x.map());
  return create_vector_wrap(*x.map(), x.bs(), x.array());
}

/// @todo This function could take just the local sizes
///
/// Compute PETSc IndexSets (IS) for a stack of index maps. E.g., if
/// `map[0] = {0, 1, 2, 3, 4, 5, 6}` and `map[1] = {0, 1, 2, 4}` (in
/// local indices) then `IS[0] = {0, 1, 2, 3, 4, 5, 6}` and `IS[1] = {7,
/// 8, 9, 10}`.
///
/// @note The caller is responsible for destruction of each IS.
///
/// @param[in] maps Vector of IndexMaps and corresponding block sizes
/// @returns Vector of PETSc Index Sets, created on` PETSC_COMM_SELF`
std::vector<IS> create_index_sets(
    const std::vector<
        std::pair<std::reference_wrapper<const common::IndexMap>, int>>& maps);

/// Copy blocks from Vec into local vectors
std::vector<std::vector<PetscScalar>> get_local_vectors(
    const Vec x,
    const std::vector<
        std::pair<std::reference_wrapper<const common::IndexMap>, int>>& maps);

/// Scatter local vectors to Vec
void scatter_local_vectors(
    Vec x, const std::vector<std::span<const PetscScalar>>& x_b,
    const std::vector<
        std::pair<std::reference_wrapper<const common::IndexMap>, int>>& maps);

/// Create a PETSc Mat. Caller is responsible for destroying the
/// returned object.
Mat create_matrix(MPI_Comm comm, const SparsityPattern& sp,
                  const std::string& type = std::string());

/// Create PETSc MatNullSpace. Caller is responsible for destruction
/// returned object.
/// @param [in] comm The MPI communicator
/// @param[in] basis The nullspace basis vectors
/// @return A PETSc nullspace object
MatNullSpace create_nullspace(MPI_Comm comm, std::span<const Vec> basis);

/// These class provides static functions that permit users to set and
/// retrieve PETSc options via the PETSc option/parameter system. The
/// option must not be prefixed by '-', e.g.
///
///     la::petsc::options::set("mat_mumps_icntl_14", 40);
namespace options
{
/// Set PETSc option that takes no value
void set(std::string option);

/// Generic function for setting PETSc option
template <typename T>
void set(std::string option, const T value)
{
  if (option[0] != '-')
    option = '-' + option;

  PetscErrorCode ierr;
  ierr = PetscOptionsSetValue(nullptr, option.c_str(),
                              boost::lexical_cast<std::string>(value).c_str());
  if (ierr != 0)
    petsc::error(ierr, __FILE__, "PetscOptionsSetValue");
}

/// Clear a PETSc option
void clear(std::string option);

/// Clear PETSc global options database
void clear();
} // namespace options

/// A simple wrapper for a PETSc vector pointer (Vec). Its main purpose
/// is to assist with memory/lifetime management of PETSc Vec objects.
///
/// Access the underlying PETSc Vec pointer using the function
/// Vector::vec() and use the full PETSc interface.
class Vector
{
public:
  /// Create a vector
  /// @note Collective
  /// @param[in] map Index map describing the parallel layout
  /// @param[in] bs the block size
  Vector(const common::IndexMap& map, int bs);

  // Delete copy constructor to avoid accidental copying of 'heavy' data
  Vector(const Vector& x) = delete;

  /// Move constructor
  Vector(Vector&& x);

  /// Create holder of a PETSc Vec object/pointer. The Vec x object
  /// should already be created. If inc_ref_count is true, the reference
  /// counter of the Vec object will be increased. The Vec reference
  /// count will always be decreased upon destruction of the
  /// PETScVector.
  ///
  /// @note Collective
  ///
  /// @param[in] x The PETSc Vec
  /// @param[in] inc_ref_count True if the reference count of `x` should
  /// be incremented
  Vector(Vec x, bool inc_ref_count);

  /// Destructor
  virtual ~Vector();

  // Assignment operator (disabled)
  Vector& operator=(const Vector& x) = delete;

  /// Move Assignment operator
  Vector& operator=(Vector&& x);

  /// Create a copy of the vector
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
  void set_options_prefix(std::string options_prefix);

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

/// This class is a base class for matrices that can be used in
/// petsc::KrylovSolver.
class Operator
{
public:
  /// Constructor
  Operator(Mat A, bool inc_ref_count);

  // Copy constructor (deleted)
  Operator(const Operator& A) = delete;

  /// Move constructor
  Operator(Operator&& A);

  /// Destructor
  virtual ~Operator();

  /// Assignment operator (deleted)
  Operator& operator=(const Operator& A) = delete;

  /// Move assignment operator
  Operator& operator=(Operator&& A);

  /// Return number of rows and columns (num_rows, num_cols). PETSc
  /// returns -1 if size has not been set.
  std::array<std::int64_t, 2> size() const;

  /// Initialize vector to be compatible with the matrix-vector product
  /// y = Ax. In the parallel case, size and layout are both important.
  ///
  /// @param[in] dim The dimension (axis): dim = 0 --> z = y, dim = 1
  /// --> z = x
  Vec create_vector(std::size_t dim) const;

  /// Return PETSc Mat pointer
  Mat mat() const;

protected:
  // PETSc Mat pointer
  Mat _matA;
};

/// It is a simple wrapper for a PETSc matrix pointer (Mat). Its main
/// purpose is to assist memory management of PETSc Mat objects.
///
/// For advanced usage, access the PETSc Mat pointer using the function
/// mat() and use the standard PETSc interface.
class Matrix : public Operator
{
public:
  /// Return a function with an interface for adding or inserting values
  /// into the matrix A (calls MatSetValuesLocal)
  /// @param[in] A The matrix to set values in
  /// @param[in] mode The PETSc insert mode (ADD_VALUES, INSERT_VALUES, ...)
  static auto set_fn(Mat A, InsertMode mode)
  {
    return [A, mode, cache = std::vector<PetscInt>()](
               const std::span<const std::int32_t>& rows,
               const std::span<const std::int32_t>& cols,
               const std::span<const PetscScalar>& vals) mutable -> int
    {
      PetscErrorCode ierr;
#ifdef PETSC_USE_64BIT_INDICES
      cache.resize(rows.size() + cols.size());
      std::copy(rows.begin(), rows.end(), cache.begin());
      std::copy(cols.begin(), cols.end(),
                std::next(cache.begin(), rows.size()));
      const PetscInt* _rows = cache.data();
      const PetscInt* _cols = cache.data() + rows.size();
      ierr = MatSetValuesLocal(A, rows.size(), _rows, cols.size(), _cols,
                               vals.data(), mode);
#else
      ierr = MatSetValuesLocal(A, rows.size(), rows.data(), cols.size(),
                               cols.data(), vals.data(), mode);
#endif

#ifndef NDEBUG
      if (ierr != 0)
        petsc::error(ierr, __FILE__, "MatSetValuesLocal");
#endif
      return ierr;
    };
  }

  /// Return a function with an interface for adding or inserting values
  /// into the matrix A using blocked indices
  /// (calls MatSetValuesBlockedLocal)
  /// @param[in] A The matrix to set values in
  /// @param[in] mode The PETSc insert mode (ADD_VALUES, INSERT_VALUES, ...)
  static auto set_block_fn(Mat A, InsertMode mode)
  {
    return [A, mode, cache = std::vector<PetscInt>()](
               const std::span<const std::int32_t>& rows,
               const std::span<const std::int32_t>& cols,
               const std::span<const PetscScalar>& vals) mutable -> int
    {
      PetscErrorCode ierr;
#ifdef PETSC_USE_64BIT_INDICES
      cache.resize(rows.size() + cols.size());
      std::copy(rows.begin(), rows.end(), cache.begin());
      std::copy(cols.begin(), cols.end(),
                std::next(cache.begin(), rows.size()));
      const PetscInt* _rows = cache.data();
      const PetscInt* _cols = cache.data() + rows.size();
      ierr = MatSetValuesBlockedLocal(A, rows.size(), _rows, cols.size(), _cols,
                                      vals.data(), mode);
#else
      ierr = MatSetValuesBlockedLocal(A, rows.size(), rows.data(), cols.size(),
                                      cols.data(), vals.data(), mode);
#endif

#ifndef NDEBUG
      if (ierr != 0)
        petsc::error(ierr, __FILE__, "MatSetValuesBlockedLocal");
#endif
      return ierr;
    };
  }

  /// Return a function with an interface for adding or inserting blocked
  /// values to the matrix A using non-blocked insertion (calls
  /// MatSetValuesLocal). Internally it expands the blocked indices into
  /// non-blocked arrays.
  /// @param[in] A The matrix to set values in
  /// @param[in] bs0 Block size for the matrix rows
  /// @param[in] bs1 Block size for the matrix columns
  /// @param[in] mode The PETSc insert mode (ADD_VALUES, INSERT_VALUES, ...)
  static auto set_block_expand_fn(Mat A, int bs0, int bs1, InsertMode mode)
  {
    return [A, bs0, bs1, mode, cache0 = std::vector<PetscInt>(),
            cache1 = std::vector<PetscInt>()](
               const std::span<const std::int32_t>& rows,
               const std::span<const std::int32_t>& cols,
               const std::span<const PetscScalar>& vals) mutable -> int
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
      if (ierr != 0)
        petsc::error(ierr, __FILE__, "MatSetValuesLocal");
#endif
      return ierr;
    };
  }

  /// Create holder for a PETSc Mat object from a sparsity pattern
  Matrix(MPI_Comm comm, const SparsityPattern& sp,
         const std::string& type = std::string());

  /// Create holder of a PETSc Mat object/pointer. The Mat A object
  /// should already be created. If inc_ref_count is true, the reference
  /// counter of the Mat will be increased. The Mat reference count will
  /// always be decreased upon destruction of the petsc::Matrix.
  Matrix(Mat A, bool inc_ref_count);

  // Copy constructor (deleted)
  Matrix(const Matrix& A) = delete;

  /// Move constructor (falls through to base class move constructor)
  Matrix(Matrix&& A) = default;

  /// Destructor
  ~Matrix() = default;

  /// Assignment operator (deleted)
  Matrix& operator=(const Matrix& A) = delete;

  /// Move assignment operator
  Matrix& operator=(Matrix&& A) = default;

  /// Assembly type
  ///   FINAL - corresponds to PETSc MAT_FINAL_ASSEMBLY
  ///   FLUSH - corresponds to PETSc MAT_FLUSH_ASSEMBLY
  enum class AssemblyType : std::int32_t
  {
    FINAL,
    FLUSH
  };

  /// Finalize assembly of tensor. The following values are recognized
  /// for the mode parameter:
  /// @param type
  ///   FINAL    - corresponds to PETSc MatAssemblyBegin+End(MAT_FINAL_ASSEMBLY)
  ///   FLUSH  - corresponds to PETSc MatAssemblyBegin+End(MAT_FLUSH_ASSEMBLY)
  void apply(AssemblyType type);

  /// Return norm of matrix
  double norm(Norm norm_type) const;

  //--- Special PETSc Functions ---

  /// Sets the prefix used by PETSc when searching the options
  /// database
  void set_options_prefix(std::string options_prefix);

  /// Returns the prefix used by PETSc when searching the options
  /// database
  std::string get_options_prefix() const;

  /// Call PETSc function MatSetFromOptions on the PETSc Mat object
  void set_from_options();
};

/// This class implements Krylov methods for linear systems of the form
/// Ax = b. It is a wrapper for the Krylov solvers of PETSc.
class KrylovSolver
{
public:
  /// Create Krylov solver for a particular method and named
  /// preconditioner
  explicit KrylovSolver(MPI_Comm comm);

  /// Create solver wrapper of a PETSc KSP object
  /// @param[in] ksp The PETSc KSP object. It should already have been created
  /// @param[in] inc_ref_count Increment the reference count on `ksp` if true
  KrylovSolver(KSP ksp, bool inc_ref_count);

  // Copy constructor (deleted)
  KrylovSolver(const KrylovSolver& solver) = delete;

  /// Move constructor
  KrylovSolver(KrylovSolver&& solver);

  /// Destructor
  ~KrylovSolver();

  // Assignment operator (deleted)
  KrylovSolver& operator=(const KrylovSolver&) = delete;

  /// Move assignment
  KrylovSolver& operator=(KrylovSolver&& solver);

  /// Set operator (Mat)
  void set_operator(const Mat A);

  /// Set operator and preconditioner matrix (Mat)
  void set_operators(const Mat A, const Mat P);

  /// Solve linear system Ax = b and return number of iterations (A^t x
  /// = b if transpose is true)
  int solve(Vec x, const Vec b, bool transpose = false) const;

  /// Sets the prefix used by PETSc when searching the PETSc options
  /// database
  void set_options_prefix(std::string options_prefix);

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
