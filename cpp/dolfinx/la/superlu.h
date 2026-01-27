// Copyright (C) 2026 Chris N. Richardson
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
/// Solver using SuperLU-dist
template <typename T>
class SuperLUSolver
{
public:
  /// @brief SuperLU-dist solver wrapper
  /// @param Amat Assembled matrix to solve for
  /// @param verbose Verbosity
  SuperLUSolver(std::shared_ptr<const dolfinx::la::MatrixCSR<T>> Amat,
                bool verbose = false);

  ~SuperLUSolver();

  /// Solve A.u=b
  /// @param b RHS Vector
  /// @param u Solution Vector
  /// @note Must be compatible with A
  int solve(const dolfinx::la::Vector<T>& b, dolfinx::la::Vector<T>& u);

private:
  /// Set the matrix operator
  void set_operator(const la::MatrixCSR<T>& Amat);

  // Pointer to struct gridinfo_t
  void* _grid;
  // Pointer to SuperMatrix
  void* _A;

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
