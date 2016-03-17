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
//
// First added:  2005-02-04

#ifndef __DOLFIN_EIGEN_KRYLOV_SOLVER_H
#define __DOLFIN_EIGEN_KRYLOV_SOLVER_H

#include <map>
#include <memory>
#include <dolfin/common/types.h>
#include "GenericLinearSolver.h"

namespace dolfin
{

  /// Forward declarations
  class EigenMatrix;
  class EigenVector;
  class GenericMatrix;
  class GenericVector;

  /// This class implements Krylov methods for linear systems of the
  /// form Ax = b. It is a wrapper for the Krylov solvers of Eigen.

  class EigenKrylovSolver : public GenericLinearSolver
  {
  public:

    /// Create Krylov solver for a particular method and names
    /// preconditioner
    EigenKrylovSolver(std::string method="default",
                      std::string preconditioner="default");

    /// Destructor
    ~EigenKrylovSolver();

    /// Set operator (matrix)
    void set_operator(std::shared_ptr<const GenericLinearOperator> A);

    /// Set operator (matrix)
    void set_operator(std::shared_ptr<const EigenMatrix> A);

    /// Set operator (matrix) and preconditioner matrix
    void set_operators(std::shared_ptr<const GenericLinearOperator> A,
                       std::shared_ptr<const GenericLinearOperator> P);

    /// Set operator (matrix) and preconditioner matrix
    void set_operators(std::shared_ptr<const EigenMatrix> A,
                       std::shared_ptr<const EigenMatrix> P);

    /// Get operator (matrix)
    const EigenMatrix& get_operator() const;

    /// Solve linear system Ax = b and return number of iterations
    std::size_t solve(GenericVector& x, const GenericVector& b);

    /// Solve linear system Ax = b and return number of iterations
    std::size_t solve(EigenVector& x, const EigenVector& b);

    /// Solve linear system Ax = b and return number of iterations
    std::size_t solve(const GenericLinearOperator& A, GenericVector& x,
                      const GenericVector& b);

    /// Solve linear system Ax = b and return number of iterations
    std::size_t solve(const EigenMatrix& A, EigenVector& x,
                      const EigenVector& b);

    /// Return informal string representation (pretty-print)
    std::string str(bool verbose) const;

    /// Return a list of available solver methods
    static std::map<std::string, std::string> methods();

    /// Return a list of available preconditioners
    static std::map<std::string, std::string> preconditioners();

    /// Default parameter values
    static Parameters default_parameters();

    /// Return parameter type: "krylov_solver" or "lu_solver"
    std::string parameter_type() const
    {
      return "krylov_solver";
    }

  private:

    // Initialize solver
    void init(const std::string method, const std::string pc="default");

    // Call with an actual solver
    template <typename Solver>
    std::size_t call_solver(Solver& solver, GenericVector& x,
                            const GenericVector& b);

    // Chosen Krylov method
    std::string _method;

    // Chosen Eigen precondtioner method
    std::string _pc;

    // Available solvers and preconditioner descriptions
    static const std::map<std::string, std::string> _methods_descr;
    static const std::map<std::string, std::string> _pcs_descr;

    // Operator (the matrix)
    std::shared_ptr<const EigenMatrix> _matA;

    // Matrix used to construct the preconditioner
    std::shared_ptr<const EigenMatrix> _matP;

    // Prepare parameters; this cannot be done in static update_parameters
    // as it depends on the method
    void _init_parameters();

    // Compute tolerance to be passed to Eigen
    double _compute_tolerance(const EigenMatrix& A, const EigenVector& x,
                              const EigenVector& b) const;
  };

}

#endif
