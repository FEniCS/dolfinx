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
//
// Modified by Steven Vandekerckhove, 2014.

#ifndef __LOCAL_SOLVER_H
#define __LOCAL_SOLVER_H

#include <memory>
#include <vector>
#include <Eigen/Dense>

namespace dolfin
{

  /// This class solves problems cell-wise. It computes the local
  /// left-hand side A_local and the local right-hand side b_local
  /// (for one cell), and solves A_local x_local = b_local. The result
  /// x_local is copied into the global vector x. The operator A_local
  /// must be square.
  ///
  /// For forms with no coupling across cell edges, this function is
  /// identical to a global solve. For problems with coupling across
  /// cells it is not.
  ///
  /// This class can be used for post-processing solutions, e.g. computing
  /// stress fields for visualisation, far more cheaply that using
  /// global projections.

  // Forward declarations
  class Form;
  class GenericVector;

  class LocalSolver
  {
  public:

    /// Constructor
    //LocalSolver();

    /// Constructor
    //LocalSolver(const Form& a, const Form& L);

    /// Constructor (shared pointer version)
    LocalSolver(std::shared_ptr<const Form> a,
                std::shared_ptr<const Form> L);

    /// Constructor
    //LocalSolver(std::shared_ptr<const Form> a);

    /// Solve local (cell-wise) problem and copy result into global
    /// vector x, reusing factorizations of local matrices.
    void solve(GenericVector& x) const;

    /// Solve local (cell-wise) problem and copy result into global
    /// vector x, reusing factorizations of local matrices.
    //void solve(GenericVector& x, const GenericVector& b) const;

    /// Solve local (cell-wise) problem and copy result into global
    /// vector x.
    /*
    void solve(GenericVector& x, const Form& a, const Form& L,
               bool symmetric=false) const;
    */

    /// Cache the LU factorisation for local matrices for re-use
    void cache_factorization(bool cache);

    /// Reset (clear) any stored factorisations
    void reset_factorization();


  private:

    // Check forms
    void check_forms() const;

    // Assemble RHS matrices
    //void init();

    // Factorise RHS for all all
    void factorize();

    // Bilinear and linear forms
    std::shared_ptr<const Form> _a, _L;

    // Cached LU factorisations of matrices
    std::vector<Eigen::PartialPivLU<Eigen::Matrix<double, Eigen::Dynamic,
      Eigen::Dynamic, Eigen::RowMajor>>> _lus;
  };

}

#endif
