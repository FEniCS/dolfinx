// Copyright (C) 2004-2018 Johan Hoffman, Johan Jansson, Anders Logg and
// Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "PETScVector.h"
#include "utils.h"
#include <boost/lexical_cast.hpp>
#include <functional>
#include <petscksp.h>
#include <petscmat.h>
#include <petscoptions.h>
#include <string>
#include <vector>
#include <xtl/xspan.hpp>

namespace dolfinx::la
{
class SparsityPattern;

namespace petsc
{
/// Create a PETSc Mat. Caller is responsible for destroying the
/// returned object.
Mat create_matrix(MPI_Comm comm, const SparsityPattern& sp,
                  const std::string& type = std::string());

/// Create PETSc MatNullSpace. Caller is responsible for destruction
/// returned object.
/// @param [in] comm The MPI communicator
/// @param[in] basis The nullspace basis vectors
/// @return A PETSc nullspace object
MatNullSpace create_nullspace(MPI_Comm comm, const xtl::span<const Vec>& basis);

/// These class provides static functions that permit users to set and
/// retrieve PETSc options via the PETSc option/parameter system. The
/// option must not be prefixed by '-', e.g.
///
///     PETScOptions::set("mat_mumps_icntl_14", 40);
class Options
{
public:
  /// Set PETSc option that takes no value
  static void set(std::string option);

  /// Generic function for setting PETSc option
  template <typename T>
  static void set(std::string option, const T value)
  {
    if (option[0] != '-')
      option = '-' + option;

    PetscErrorCode ierr;
    ierr
        = PetscOptionsSetValue(nullptr, option.c_str(),
                               boost::lexical_cast<std::string>(value).c_str());
    if (ierr != 0)
      petsc::error(ierr, __FILE__, "PetscOptionsSetValue");
  }

  /// Clear a PETSc option
  static void clear(std::string option);

  /// Clear PETSc global options database
  static void clear();
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
  ///                --> z = x
  Vector create_vector(std::size_t dim) const;

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
  static std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                           const std::int32_t*, const PetscScalar*)>
  set_fn(Mat A, InsertMode mode);

  /// Return a function with an interface for adding or inserting values
  /// into the matrix A using blocked indices
  /// (calls MatSetValuesBlockedLocal)
  /// @param[in] A The matrix to set values in
  /// @param[in] mode The PETSc insert mode (ADD_VALUES, INSERT_VALUES, ...)
  static std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                           const std::int32_t*, const PetscScalar*)>
  set_block_fn(Mat A, InsertMode mode);

  /// Return a function with an interface for adding or inserting blocked
  /// values to the matrix A using non-blocked insertion (calls
  /// MatSetValuesLocal). Internally it expands the blocked indices into
  /// non-blocked arrays.
  /// @param[in] A The matrix to set values in
  /// @param[in] bs0 Block size for the matrix rows
  /// @param[in] bs1 Block size for the matrix columns
  /// @param[in] mode The PETSc insert mode (ADD_VALUES, INSERT_VALUES, ...)
  static std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                           const std::int32_t*, const PetscScalar*)>
  set_block_expand_fn(Mat A, int bs0, int bs1, InsertMode mode);

  /// Create holder for a PETSc Mat object from a sparsity pattern
  Matrix(MPI_Comm comm, const SparsityPattern& sp,
         const std::string& type = std::string());

  /// Create holder of a PETSc Mat object/pointer. The Mat A object
  /// should already be created. If inc_ref_count is true, the reference
  /// counter of the Mat will be increased. The Mat reference count will
  /// always be decreased upon destruction of the the petsc::Matrix.
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
