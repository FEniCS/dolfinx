// Copyright (C) 2026 Jack S. Hale, Chris N. Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#ifdef HAS_SUPERLU_DIST
#include <dolfinx/la/MatrixCSR.h>
#include <dolfinx/la/Vector.h>

namespace dolfinx::la
{

// Trick for foreword declaration of anonymous structs from SuperLU
// Avoids including SuperLU headers in DOLFINx headers
class SuperLUStructs
{
public:
  struct SuperMatrix;
  struct gridinfo_t;
};

/// Solver using SuperLU-dist
template <typename T>
class SuperLUSolver
{
public:
  /// @brief SuperLU_dist solver wrapper
  /// @param Amat Assembled matrix to solve for
  /// @param verbose Verbosity
  /// @tparam T Scalar type
  SuperLUSolver(std::shared_ptr<const MatrixCSR<T>> Amat, bool verbose = false);

  /// Copy constructor
  SuperLUSolver(const SuperLUSolver&) = delete;

  /// Copy assignment
  SuperLUSolver& operator=(const SuperLUSolver&) = delete;

  /// Solve linear system Au = b
  /// @param b Right-hand side Vector
  /// @param u Solution Vector
  /// @note Must be compatible with A
  int solve(const dolfinx::la::Vector<T>& b, dolfinx::la::Vector<T>& u) const;

private:
  // Call library cleanup and delete pointer. For use with std::unique_ptr
  // holding gridinfo_t.
  struct GridInfoDeleter
  {
    void operator()(SuperLUStructs::gridinfo_t* g) const noexcept;
  };

  // Call library cleanup and delete pointer. For use with std::unique_ptr
  // holding SuperMatrix.
  struct SuperMatrixDeleter
  {
    void operator()(SuperLUStructs::SuperMatrix* A) const noexcept;
  };

  /// Set the matrix operator
  void set_operator(const la::MatrixCSR<T>& Amat);

  // Pointer to struct gridinfo_t
  std::unique_ptr<SuperLUStructs::gridinfo_t, GridInfoDeleter> _gridinfo;
  // Pointer to SuperMatrix
  std::unique_ptr<SuperLUStructs::SuperMatrix, SuperMatrixDeleter> _supermatrix;

  // Saved matrix operator with rows and cols in
  // required integer type
  std::shared_ptr<const la::MatrixCSR<T>> _Amat;
  std::vector<int> cols;
  std::vector<int> rowptr;

  // Flag for diagnostic output
  bool _verbose;
};
} // namespace dolfinx::la
#endif
