// Copyright (C) 2026 Jack S. Hale, Chris N. Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#ifdef HAS_SUPERLU_DIST

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
};

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
  /// @tparam T Scalar type.
  /// @param A Matrix.
  SuperLUDistMatrix(std::shared_ptr<const MatrixCSR<T>> A);

  /// Copy constructor
  SuperLUDistMatrix(const SuperLUDistMatrix&) = delete;

  /// Copy assignment
  SuperLUDistMatrix& operator=(const SuperLUDistMatrix&) = delete;

  /// Get non-const pointer to SuperLU_DIST SuperMatrix.
  SuperLUDistStructs::SuperMatrix* supermatrix() const;

  /// Get MatrixCSR (const).
  const MatrixCSR<T>& matA() const;

private:
  // Saved matrix operator with rows and cols in required integer type.
  // cols and rowptr are required in opaque type "int_t" of
  // SuperLU_DIST.
  std::shared_ptr<const MatrixCSR<T>> _matA;
  std::unique_ptr<SuperLUDistStructs::vec_int_t> _cols;
  std::unique_ptr<SuperLUDistStructs::vec_int_t> _rowptr;

  // Pointer to native SuperMatrix
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

/// SuperLU_DIST linear solver interface.
template <typename T>
class SuperLUDistSolver
{
public:
  /// @brief Create solver for a SuperLU_DIST matrix operator.
  ///
  /// Solves Au = b using SuperLU_DIST.
  ///
  /// @tparam T Scalar type.
  /// @param A Matrix to solve for.
  /// @param verbose Verbose output.
  SuperLUDistSolver(std::shared_ptr<const SuperLUDistMatrix<T>> A);

  /// Copy constructor
  SuperLUDistSolver(const SuperLUDistSolver&) = delete;

  /// Copy assignment
  SuperLUDistSolver& operator=(const SuperLUDistSolver&) = delete;

  void set_option(std::string option, std::string value);
  void set_options(SuperLUDistStructs::superlu_dist_options_t options);

  /// @brief Solve linear system Au = b.
  ///
  /// @param b Right-hand side vector.
  /// @param u Solution vector.
  /// @returns SuperLU_DIST info flag.
  /// @note Vectors must have size and parallel layout that is
  /// compatible with `A`.
  int solve(const Vector<T>& b, Vector<T>& u) const;

private:
  // Wrapped SuperLU SuperMatrix
  std::shared_ptr<const SuperLUDistMatrix<T>> _superlu_matA;

  // Pointer to struct superlu_dist_options_t
  std::unique_ptr<SuperLUDistStructs::superlu_dist_options_t> _options;

  // Pointer to struct gridinfo_t
  std::unique_ptr<SuperLUDistStructs::gridinfo_t, GridInfoDeleter> _gridinfo;
};
} // namespace dolfinx::la
#endif
