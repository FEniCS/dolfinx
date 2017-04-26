// Copyright (C) 2015 Chris Richardson
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

#ifndef __DOLFIN_EIGEN_LU_SOLVER_H
#define __DOLFIN_EIGEN_LU_SOLVER_H

#include <map>
#include <memory>

#include <dolfin/common/types.h>
#include <Eigen/Dense>
#include "GenericLinearSolver.h"

namespace dolfin
{
  /// Forward declarations
  class EigenMatrix;
  class EigenVector;
  class GenericLinearOperator;
  class GenericVector;

  /// This class implements the direct solution (LU factorization) for
  /// linear systems of the form Ax = b.

  class EigenLUSolver : public GenericLinearSolver
  {
  public:

    /// Constructor
    EigenLUSolver(std::string method="default");

    /// Constructor
    EigenLUSolver(std::shared_ptr<const EigenMatrix> A,
                  std::string method="default");

    /// Destructor
    ~EigenLUSolver();

    /// Set operator (matrix)
    void set_operator(std::shared_ptr<const GenericLinearOperator> A);

    /// Set operator (matrix)
    void set_operator(std::shared_ptr<const EigenMatrix> A);

    /// Get operator (matrix)
    const GenericLinearOperator& get_operator() const;

    /// Solve linear system Ax = b
    std::size_t solve(GenericVector& x, const GenericVector& b);

    /// Solve linear system Ax = b
    std::size_t solve(const GenericLinearOperator& A, GenericVector& x,
                      const GenericVector& b);

    /// Solve linear system Ax = b
    std::size_t solve(const EigenMatrix& A, EigenVector& x,
                      const EigenVector& b);

    /// Return informal string representation (pretty-print)
    std::string str(bool verbose) const;

    /// Return a list of available solver methods
    static std::map<std::string, std::string> methods();

    /// Default parameter values
    static Parameters default_parameters();

  private:

    // Call generic solve
    template <typename Solver>
    void call_solver(Solver& solver, GenericVector& x, const GenericVector& b);

    // Available LU solvers and descriptions
    static const std::map<std::string, std::string> _methods_descr;

    // Current selected method
    std::string _method;

    // Select LU solver type
    std::string select_solver(const std::string method) const;

    // Operator (the matrix)
    std::shared_ptr<const EigenMatrix> _matA;

  };

}

#endif
