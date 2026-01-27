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
template <typename T>

/// Solver using SuperLU-dist
class SuperLUSolver
{
public:
  /// @brief SuperLU-dist solver wrapper
  /// @param comm MPI Communicator
  /// @param verbose Verbosity
  SuperLUSolver(MPI_Comm comm, bool verbose = false);

  ~SuperLUSolver();

  /// @brief Set the matrix operator
  /// @param Amat MatrixCSR
  void set_operator(const dolfinx::la::MatrixCSR<T>& Amat);

  /// Solve A.u=b
  /// @param b RHS Vector
  /// @param u Solution Vector
  /// @note Must be compatible with A
  int solve(const dolfinx::la::Vector<T>& b, dolfinx::la::Vector<T>& u);

private:
  // Pointer to struct gridinfo_t
  void* _grid;
  // Pointer to SuperMatrix
  void* _A;

  std::vector<int> cols;
  std::vector<int> rowptr;

  int m_loc, m;
  MPI_Comm _comm;
  bool _verbose;
};
}
#endif
