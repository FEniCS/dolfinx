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

namespace dolfinx::la
{

// Delcare structs to avoid exposing SuperLU_DIST headers in DOLFINx.
class SuperLUDistStructs
{
public:
  struct SuperMatrix;
  struct gridinfo_t;
  struct vec_int_t;
};

/// SuperLU_DIST linear solver interface.
template <typename T>
class SuperLUDistSolver
{
public:
  /// @brief Create solver for a matrix operator.
  ///
  /// Solves Au = b using SuperLU_DIST.
  ///
  /// @tparam T Scalar type.
  /// @param A Matrix to solve for.
  /// @param verbose Verbose output.
  SuperLUDistSolver(std::shared_ptr<const MatrixCSR<T>> A,
                    bool verbose = false);

  /// Copy constructor
  SuperLUDistSolver(const SuperLUDistSolver&) = delete;

  /// Copy assignment
  SuperLUDistSolver& operator=(const SuperLUDistSolver&) = delete;

  /// @brief Solve linear system Au = b.
  ///
  /// @param b Right-hand side vector.
  /// @param u Solution vector.
  /// @returns SuperLU_DIST info flag.
  /// @note Vectors must have size and parallel layout that is
  /// compatible with `A`.
  int solve(const Vector<T>& b, Vector<T>& u) const;

private:
  // Call library cleanup and delete pointer. For use with
  // std::unique_ptr holding gridinfo_t.
  struct GridInfoDeleter
  {
    void operator()(SuperLUDistStructs::gridinfo_t* g) const noexcept;
  };

  // Call library cleanup and delete pointer. For use with
  // std::unique_ptr holding SuperMatrix.
  struct SuperMatrixDeleter
  {
    void operator()(SuperLUDistStructs::SuperMatrix* A) const noexcept;
  };

  struct VecIntDeleter
  {
    void operator()(SuperLUDistStructs::vec_int_t* v) const noexcept;
  };

  // Saved matrix operator with rows and cols in required integer type.
  // cols and rowptr are required in opaque type "int_t" of
  // SuperLU_DIST.
  std::shared_ptr<const la::MatrixCSR<T>> _Amat;
  std::unique_ptr<SuperLUDistStructs::vec_int_t, VecIntDeleter> _cols;
  std::unique_ptr<SuperLUDistStructs::vec_int_t, VecIntDeleter> _rowptr;

  // Pointer to struct gridinfo_t
  std::unique_ptr<SuperLUDistStructs::gridinfo_t, GridInfoDeleter> _gridinfo;
  // Pointer to SuperMatrix
  std::unique_ptr<SuperLUDistStructs::SuperMatrix, SuperMatrixDeleter>
      _supermatrix;

  // Flag for diagnostic output
  bool _verbose;
};
} // namespace dolfinx::la
#endif
