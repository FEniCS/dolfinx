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

#ifndef __DOLFIN_AMESOS2_LU_SOLVER_H
#define __DOLFIN_AMESOS2_LU_SOLVER_H

#ifdef HAS_TRILINOS

#include "GenericLUSolver.h"

#include <Amesos2_Factory.hpp>

namespace dolfin
{
  /// Forward declarations
  class GenericLinearOperator;
  class GenericVector;
  class TpetraMatrix;
  class TpetraVector;

  /// This class implements the direct solution (LU factorization) for
  /// linear systems of the form Ax = b.
  /// It is a wrapper for the Trilinos Amesos2 LU solver.

  class Amesos2LUSolver : public GenericLUSolver
  {
  public:

    /// Constructor
    Amesos2LUSolver(std::string method="default");

    /// Constructor
    Amesos2LUSolver(std::shared_ptr<const TpetraMatrix> A,
                    std::string method="default");

    /// Destructor
    ~Amesos2LUSolver();

    /// Set operator (matrix)
    void set_operator(std::shared_ptr<const GenericLinearOperator> A);

    /// Set operator (matrix)
    void set_operator(std::shared_ptr<const TpetraMatrix> A);

    /// Get operator (matrix)
    const GenericLinearOperator& get_operator() const;

    /// Solve linear system Ax = b
    std::size_t solve(GenericVector& x, const GenericVector& b);

    /// Solve linear system Ax = b
    std::size_t solve(const GenericLinearOperator& A,
                      GenericVector& x,
                      const GenericVector& b);

    /// Solve linear system Ax = b
    std::size_t solve(const TpetraMatrix& A,
                      TpetraVector& x,
                      const TpetraVector& b);

    /// Return informal string representation (pretty-print)
    std::string str(bool verbose) const;

    /// Return a list of available solver methods
    static std::map<std::string, std::string> methods();

    /// Default parameter values
    static Parameters default_parameters();

  private:

    void init_solver(std::string& method);

    // Reference counted pointer (RCP) to solver
    Teuchos::RCP<Amesos2::Solver<TpetraMatrix::matrix_type,
                                 TpetraVector::vector_type>> _solver;

    // Operator (the matrix)
    std::shared_ptr<const TpetraMatrix> _matA;

    // Method name
    std::string _method_name;
  };

}

#endif

#endif
