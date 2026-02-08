// Copyright (C) 2026 Jack S. Hale, Chris N. Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#ifdef HAS_SUPERLU_DIST

#include <dolfinx/common/MPI.h>
#include <dolfinx/la/MatrixCSR.h>
#include <dolfinx/la/Vector.h>
#include <memory>
#include <string>

namespace dolfinx::la
{
// Declare structs to avoid exposing SuperLU_DIST headers in DOLFINx.
class SuperLUDistStructs
{
public:
  struct SuperMatrix;
  struct vec_int_t;
  struct gridinfo_t;
  struct superlu_dist_options_t;

  struct sScalePermstruct_t;
  struct dScalePermstruct_t;
  struct zScalePermstruct_t;

  struct sLUstruct_t;
  struct dLUstruct_t;
  struct zLUstruct_t;

  struct sSOLVEstruct_t;
  struct dSOLVEstruct_t;
  struct zSOLVEstruct_t;
};

// SuperLU_DIST has structs that are 'typed' with prefixes d, s, z. This allows
// the solver class to select the typed set based on T.
namespace detail
{
template <typename...>
constexpr bool always_false_v = false;

template <typename T>
struct map
{
  static_assert(
      always_false_v<T>,
      "Unsupported T. Supported: float, double, std::complex<double>.");
};

template <>
struct map<float>
{
  using ScalePermstruct_t = SuperLUDistStructs::sScalePermstruct_t;
  using LUstruct_t = SuperLUDistStructs::sLUstruct_t;
  using SOLVEstruct_t = SuperLUDistStructs::sSOLVEstruct_t;
};

template <>
struct map<double>
{
  using ScalePermstruct_t = SuperLUDistStructs::dScalePermstruct_t;
  using LUstruct_t = SuperLUDistStructs::dLUstruct_t;
  using SOLVEstruct_t = SuperLUDistStructs::dSOLVEstruct_t;
};

template <>
struct map<std::complex<double>>
{
  using ScalePermstruct_t = SuperLUDistStructs::zScalePermstruct_t;
  using LUstruct_t = SuperLUDistStructs::zLUstruct_t;
  using SOLVEstruct_t = SuperLUDistStructs::zSOLVEstruct_t;
};

} // namespace detail

template <class T>
using map_t = detail::map<T>;

/// Call library cleanup and delete pointer. For use with
/// std::unique_ptr holding SuperMatrix.
struct SuperMatrixDeleter
{
  /// @brief Deletion on SuperMatrix
  /// @param A
  void operator()(SuperLUDistStructs::SuperMatrix* A) const noexcept;
};

/// SuperLU_DIST matrix interface.
template <typename T>
class SuperLUDistMatrix
{
public:
  /// @brief Create SuperLU_DIST matrix operator.
  ///
  /// Copies data from native CSR into SuperLU_DIST format.
  ///
  /// @tparam T Scalar type.
  /// @param A Matrix.
  SuperLUDistMatrix(const MatrixCSR<T>& A);

  /// @brief Copy constructor (deleted)
  SuperLUDistMatrix(const SuperLUDistMatrix&) = delete;

  /// @brief Copy assignment (deleted)
  SuperLUDistMatrix& operator=(const SuperLUDistMatrix&) = delete;

  /// @brief Get MPI communicator that matrix is defined on.
  MPI_Comm comm() const;

  /// @brief Get pointer to SuperLU_DIST SuperMatrix (non-const).
  SuperLUDistStructs::SuperMatrix* supermatrix() const;

private:
  dolfinx::MPI::Comm _comm;
  // Deep copy of values from MatrixCSR.
  std::vector<T> _matA_values;
  // Saved matrix operator with rows and cols in required integer type.
  // cols and rowptr are required in opaque type "int_t" of
  // SuperLU_DIST.
  std::unique_ptr<SuperLUDistStructs::vec_int_t> _cols;
  std::unique_ptr<SuperLUDistStructs::vec_int_t> _rowptr;

  // Pointer to native SuperMatrix for use in solver
  std::unique_ptr<SuperLUDistStructs::SuperMatrix, SuperMatrixDeleter>
      _supermatrix;
};

/// Call library cleanup and delete pointer. For use with
/// std::unique_ptr holding gridinfo_t.
struct GridInfoDeleter
{
  /// @brief Deletion of gridinfo_t
  /// @param g
  void operator()(SuperLUDistStructs::gridinfo_t* g) const noexcept;
};

struct ScalePermStructDeleter
{
  void operator()(SuperLUDistStructs::sScalePermstruct_t* s) const noexcept;
  void operator()(SuperLUDistStructs::dScalePermstruct_t* s) const noexcept;
  void operator()(SuperLUDistStructs::zScalePermstruct_t* s) const noexcept;
};

struct LUStructDeleter
{
  void operator()(SuperLUDistStructs::sLUstruct_t* l) const noexcept;
  void operator()(SuperLUDistStructs::dLUstruct_t* l) const noexcept;
  void operator()(SuperLUDistStructs::zLUstruct_t* l) const noexcept;
};

struct SolveStructDeleter
{
  void operator()(SuperLUDistStructs::sSOLVEstruct_t* S) const noexcept;
  void operator()(SuperLUDistStructs::dSOLVEstruct_t* S) const noexcept;
  void operator()(SuperLUDistStructs::zSOLVEstruct_t* S) const noexcept;
};

/// SuperLU_DIST linear solver interface.
template <typename T>
class SuperLUDistSolver
{
public:
  /// @brief Create solver for a SuperLU_DIST matrix operator.
  ///
  /// Solves linear system Au = b via LU decomposition.
  ///
  /// The SuperLU_DIST solver has options set to upstream defaults,
  /// except PrintStat (verbose solver output) set to NO.
  ///
  /// @tparam T Scalar type.
  /// @param A Assembled left-hand side matrix.
  SuperLUDistSolver(std::shared_ptr<const SuperLUDistMatrix<T>> A);

  /// Copy constructor
  SuperLUDistSolver(const SuperLUDistSolver&) = delete;

  /// Copy assignment
  SuperLUDistSolver& operator=(const SuperLUDistSolver&) = delete;

  /// @brief Set solver option (name, value)
  ///
  /// See SuperLU_DIST User's Guide for option names and values.
  ///
  /// @param name Option name.
  /// @param value Option value.
  void set_option(std::string name, std::string value);

  /// @brief Set all solver options (native struct)
  ///
  /// See SuperLU_DIST User's Guide for option names and values.
  ///
  /// Callers must complete the forward declared struct, e.g.:
  ///
  /// ```cpp
  /// #include <superlu_defs.h>
  /// struct dolfinx::la::SuperLUDistStructs::superlu_dist_options_t
  ///  : public ::superlu_dist_options_t
  /// {
  /// };
  ///
  /// SuperLUDistStructs::superlu_dist_options_t options;
  /// set_default_options_dist(&options);
  /// options.PrintStat = YES;
  /// // Setup SuperLUDistMatrix and SuperLUDistSolver
  /// solver.set_options(options);
  /// ```
  ///
  /// @param options SuperLU_DIST option struct.
  void set_options(SuperLUDistStructs::superlu_dist_options_t options);

  /// @brief Solve linear system Au = b.
  ///
  /// @param b Right-hand side vector.
  /// @param u Solution vector, overwritten during solve.
  /// @returns SuperLU_DIST info integer.
  /// @note The caller must check the return code for success `(== 0)`.
  /// @note The caller must `u.scatter_forward()` after the solve.
  /// @note Vectors must have size and parallel layout compatible with `A`.
  int solve(const Vector<T>& b, Vector<T>& u) const;

private:
  // Assembled left-hand side matrix
  std::shared_ptr<const SuperLUDistMatrix<T>> _superlu_matA;

  // Pointer to struct superlu_dist_options_t
  std::unique_ptr<SuperLUDistStructs::superlu_dist_options_t> _options;

  // Pointer to struct gridinfo_t
  std::unique_ptr<SuperLUDistStructs::gridinfo_t, GridInfoDeleter> _gridinfo;

  // Pointer to 'typed' struct *ScalePermstruct_t
  std::unique_ptr<typename map_t<T>::ScalePermstruct_t, ScalePermStructDeleter>
      _scalepermstruct;
  // Pointer to 'typed' struct *LUstruct_t
  std::unique_ptr<typename map_t<T>::LUstruct_t, LUStructDeleter> _lustruct;
  // Pointer to 'typed' struct *SOLVEstruct
  std::unique_ptr<typename map_t<T>::SOLVEstruct_t> _solvestruct;
};
} // namespace dolfinx::la
#endif
