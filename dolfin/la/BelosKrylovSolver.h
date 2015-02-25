// Copyright (C) 2014 Chris Richardson
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

#ifndef __DOLFIN_BELOS_KRYLOV_SOLVER_H
#define __DOLFIN_BELOS_KRYLOV_SOLVER_H

#ifdef HAS_TRILINOS

#include <BelosTpetraAdapter.hpp>
#include <BelosSolverFactory.hpp>

#include <map>
#include <memory>
#include <dolfin/common/types.h>
#include "GenericLinearSolver.h"

#include "TpetraVector.h"
#include "TpetraMatrix.h"

typedef Tpetra::Operator<scalar_type, local_ordinal_type,
                         global_ordinal_type, node_type> op_type;

typedef Tpetra::MultiVector<scalar_type, local_ordinal_type,
                            global_ordinal_type, node_type> mv_type;

typedef Belos::LinearProblem<scalar_type, mv_type, op_type> problem_type;


namespace dolfin
{

  /// Forward declarations
  class GenericMatrix;
  class GenericVector;
  class TpetraMatrix;
  class TpetraVector;

  /// This class implements Krylov methods for linear systems
  /// of the form Ax = b. It is a wrapper for the Belos iterative solver
  /// from Trilinos

  class BelosKrylovSolver : public GenericLinearSolver
  {
  public:

    /// Create Krylov solver for a particular method and names
    /// preconditioner
    BelosKrylovSolver(std::string method = "default",
                      std::string preconditioner = "default");

    /// Create Krylov solver for a particular method and
    /// BelosPreconditioner
    //    BelosKrylovSolver(std::string method, BelosPreconditioner& preconditioner);

    /// Create Krylov solver for a particular method and
    /// BelosPreconditioner (shared_ptr version)
    //    BelosKrylovSolver(std::string method,
    //		      std::shared_ptr<BelosPreconditioner> preconditioner);

    /// Create Krylov solver for a particular method and
    /// BelosPreconditioner
    //    BelosKrylovSolver(std::string method,
    //                      BelosUserPreconditioner& preconditioner);

    /// Create Krylov solver for a particular method and
    /// BelosPreconditioner (shared_ptr version)
    //    BelosKrylovSolver(std::string method,
    //		    std::shared_ptr<BelosUserPreconditioner> preconditioner);

    /// Create solver wrapper of a RCP object
    //    explicit BelosKrylovSolver(Teuchos::RCP<Belos::SolverManager>);

    /// Destructor
    ~BelosKrylovSolver();

    /// Set operator (matrix)
    void set_operator(std::shared_ptr<const GenericLinearOperator> A);

    /// Set operator (matrix)
    void set_operator(std::shared_ptr<const TpetraMatrix> A);

    /// Set operator (matrix) and preconditioner matrix
    void set_operators(std::shared_ptr<const GenericLinearOperator> A,
                       std::shared_ptr<const GenericLinearOperator> P);

    /// Set operator (matrix) and preconditioner matrix
    void set_operators(std::shared_ptr<const TpetraMatrix> A,
                       std::shared_ptr<const TpetraMatrix> P);

    /// Set null space of the operator (matrix). This is used to solve
    /// singular systems
    //    void set_nullspace(const VectorSpaceBasis& nullspace);

    /// Get operator (matrix)
    const TpetraMatrix& get_operator() const;

    /// Solve linear system Ax = b and return number of iterations
    std::size_t solve(GenericVector& x, const GenericVector& b);

    /// Solve linear system Ax = b and return number of iterations
    std::size_t solve(TpetraVector& x, const TpetraVector& b);

    /// Solve linear system Ax = b and return number of iterations
    std::size_t solve(const GenericLinearOperator& A, GenericVector& x,
                      const GenericVector& b);

    /// Solve linear system Ax = b and return number of iterations
    std::size_t solve(const TpetraMatrix& A, TpetraVector& x,
                      const TpetraVector& b);

    /// Return informal string representation (pretty-print)
    std::string str(bool verbose) const;

    /// Return Belos pointer
    //    Teuchos::RCP<Belos::SolverManager<> > ksp() const;

    /// Return a list of available solver methods
    static std::map<std::string, std::string> methods();

    /// Return a list of available preconditioners
    static std::map<std::string, std::string> preconditioners();

    ///// Set options prefix
    //void set_options_prefix(std::string prefix);

    /// Default parameter values
    static Parameters default_parameters();

  private:

    // Initialize solver
    void init(const std::string& method);

    // Set options for solver
    void set_options();

    void check_dimensions(const TpetraMatrix& A, const GenericVector& x,
                          const GenericVector& b) const;

    // Belos solver pointer
    Teuchos::RCP<Belos::SolverManager<scalar_type, mv_type, op_type> >
      _solver;

    Teuchos::RCP<problem_type> _problem;

    // Operator (the matrix)
    std::shared_ptr<const TpetraMatrix> _matA;

    // Matrix used to construct the preconditioner
    std::shared_ptr<const TpetraMatrix> _matP;

  };

}

#endif

#endif
