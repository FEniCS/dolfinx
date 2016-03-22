// Copyright (C) 2013-2015 Garth N. Wells
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.

#ifndef __LOCAL_SOLVER_H
#define __LOCAL_SOLVER_H

#include <memory>
#include <vector>
#include <Eigen/Cholesky>
#include <Eigen/Dense>
#include <Eigen/LU>

namespace dolfin
{

  /// This class solves problems cell-wise. It computes the local
  /// left-hand side A_local which must be square locally but not
  /// globally. The right-hand side b_local is either computed locally
  /// for one cell or globally for all cells depending on which of the
  /// solve_local_rhs or solve_global_rhs methods which are
  /// called. You can optionally assemble the right-hand side vector
  /// yourself and use the solve_local method. You must then provide
  /// the DofMap of the right-hand side.
  ///
  /// The local solver solves A_local x_local = b_local. The result
  /// x_local is copied into the global vector x of the provided
  /// Function u. You can optionally call the factorize() method to
  /// pre-calculate the local left-hand side factorizations to speed
  /// up repeated applications of the LocalSolver with the same
  /// LHS. The solve_xxx methods will factorise the LHS A_local
  /// matrices each time if they are not cached by a previous call to
  /// factorize. You can chose upon initialization whether you want
  /// Cholesky or LU (default) factorisations.
  ///
  /// For forms with no coupling across cell edges, this function is
  /// identical to a global solve. For problems with coupling across
  /// cells it is not.
  ///
  /// This class can be used for post-processing solutions,
  /// e.g. computing stress fields for visualisation, far more cheaply
  /// that using global projections.

  // Forward declarations
  class Form;
  class Function;
  class GenericDofMap;
  class GenericVector;

  class LocalSolver
  {
  public:

    enum class SolverType {LU, Cholesky};

    /// Constructor (shared pointer version)
    LocalSolver(std::shared_ptr<const Form> a, std::shared_ptr<const Form> L,
                SolverType solver_type=SolverType::LU);

    /// Constructor (shared pointer version)
    LocalSolver(std::shared_ptr<const Form> a, SolverType solver_type=SolverType::LU);

    /// Solve local (cell-wise) problems A_e x_e = b_e, where A_e is
    /// the cell matrix LHS and b_e is the global RHS vector b
    /// restricted to the cell, i.e. b_e may contain contributions
    /// from neighbouring cells. The solution is exact for the case in
    /// which there is no coupling between cell contributions to the
    /// global matrix A, e.g. the discontinuous Galerkin matrix. The
    /// result is copied into x.
    void solve_global_rhs(Function& u) const;

    /// Solve local (cell-wise) problems A_e x_e = b_e where A_e and
    /// b_e are the cell element tensors. Compared to solve_global_rhs
    /// this function calculates local RHS vectors for each cell and
    /// hence does not include contributions from neighbouring cells.
    ///
    /// This function is useful for computing (approximate) cell-wise
    /// projections, for example for post-processing. It much more
    /// efficient than computing global projections.
    void solve_local_rhs(Function& u) const;

    /// Solve local problems for given RHS and corresponding dofmap
    /// for RHS
    void solve_local(GenericVector& x, const GenericVector& b,
                     const GenericDofMap& dofmap_b) const;

    /// Factorise the local LHS matrices for all cells and store in cache
    void factorize();

    /// Reset (clear) any stored factorizations
    void clear_factorization();

  private:

    // Bilinear and linear forms
    std::shared_ptr<const Form> _a, _formL;

    // Solver type to use
    const SolverType _solver_type;

    // Cached LU factorisations of matrices (_spd==false)
    std::vector<Eigen::PartialPivLU<Eigen::Matrix<double, Eigen::Dynamic,
                                                  Eigen::Dynamic,
                                                  Eigen::RowMajor>>> _lu_cache;

    // Cached Cholesky factorisations of matrices (_spd==true)
    std::vector<Eigen::LLT<Eigen::Matrix<double, Eigen::Dynamic,
                                         Eigen::Dynamic,
                                         Eigen::RowMajor>>> _cholesky_cache;

    // Helper function that does the actual calculations
    void _solve_local(GenericVector& x,
                      const GenericVector* global_b,
                      const GenericDofMap* dofmap_L) const;
  };

}

#endif
