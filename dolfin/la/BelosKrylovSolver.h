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

#include <map>
#include <memory>

#include <BelosTpetraAdapter.hpp>
#include <BelosSolverFactory.hpp>
#include <Ifpack2_Factory.hpp>

#include <dolfin/common/types.h>
#include "GenericLinearSolver.h"
#include "TpetraVector.h"
#include "TpetraMatrix.h"
#include "TrilinosPreconditioner.h"

namespace dolfin
{

  /// Forward declarations
  class GenericMatrix;
  class GenericVector;
  class TpetraMatrix;
  class TpetraVector;
  class TrilinosPreconditioner;

  /// This class implements Krylov methods for linear systems
  /// of the form Ax = b. It is a wrapper for the Belos iterative solver
  /// from Trilinos

  class BelosKrylovSolver : public GenericLinearSolver
  {
  public:

    typedef Tpetra::Operator<double, int, dolfin::la_index,
                             TpetraVector::node_type> op_type;
    typedef Belos::LinearProblem<double, TpetraVector::vector_type,
                                 op_type> problem_type;

    /// Create Krylov solver for a particular method and names
    /// preconditioner
    BelosKrylovSolver(std::string method = "default",
                      std::string preconditioner = "default");

    /// Create Krylov solver for a particular method and TrilinosPreconditioner
    BelosKrylovSolver(std::string method,
                      std::shared_ptr<TrilinosPreconditioner> preconditioner);

    /// Destructor
    ~BelosKrylovSolver();

    /// Set operator (matrix)
    void set_operator(std::shared_ptr<const GenericLinearOperator> A);

    /// Set operator (matrix) and preconditioner matrix
    void set_operators(std::shared_ptr<const GenericLinearOperator> A,
                       std::shared_ptr<const GenericLinearOperator> P);

    /// Set null space of the operator (matrix). This is used to solve
    /// singular systems
    //    void set_nullspace(const VectorSpaceBasis& nullspace);

    /// Get operator (matrix)
    const TpetraMatrix& get_operator() const;

    /// Solve linear system Ax = b and return number of iterations
    std::size_t solve(GenericVector& x, const GenericVector& b);

    /// Solve linear system Ax = b and return number of iterations
    std::size_t solve(const GenericLinearOperator& A, GenericVector& x,
                      const GenericVector& b);

    /// Return informal string representation (pretty-print)
    std::string str(bool verbose) const;

    /// Return Belos pointer
    //    Teuchos::RCP<Belos::SolverManager<>> solver_manager() const;

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

    friend class Ifpack2Preconditioner;
    friend class MueluPreconditioner;

    // Initialize solver
    void init(const std::string& method);

    // Set operator (matrix)
    void _set_operator(std::shared_ptr<const TpetraMatrix> A);

    // Set operator (matrix) and preconditioner matrix
    void _set_operators(std::shared_ptr<const TpetraMatrix> A,
                        std::shared_ptr<const TpetraMatrix> P);

    // Solve linear system Ax = b and return number of iterations
    std::size_t _solve(TpetraVector& x, const TpetraVector& b);

    // Solve linear system Ax = b and return number of iterations
    std::size_t _solve(const TpetraMatrix& A, TpetraVector& x,
                       const TpetraVector& b);

    // Set options for solver
    void set_options();

    void check_dimensions(const TpetraMatrix& A, const GenericVector& x,
                          const GenericVector& b) const;

    // Belos solver pointer
    Teuchos::RCP<Belos::SolverManager<double, TpetraVector::vector_type,
                                      op_type>> _solver;

    // The preconditioner, if any
    std::shared_ptr<TrilinosPreconditioner> _prec;

    // Container for the problem, see Belos::LinearProblem
    // documentation
    Teuchos::RCP<problem_type> _problem;

    // Operator (the matrix)
    std::shared_ptr<const TpetraMatrix> _matA;

  };

}

#endif

#endif
