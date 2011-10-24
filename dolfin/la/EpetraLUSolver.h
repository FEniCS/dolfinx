// Copyright (C) 2008-2010 Kent-Andre Mardal and Garth N. Wells
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
// Modified by Anders Logg 2011
//
// First added:  2008
// Last changed: 2011-10-19

#ifdef HAS_TRILINOS

#ifndef __EPETRA_LU_SOLVER_H
#define __EPETRA_LU_SOLVER_H

#include <boost/scoped_ptr.hpp>
#include "GenericLUSolver.h"

/// Forward declaration
class Amesos_BaseSolver;
class Epetra_LinearProblem;

namespace dolfin
{
  /// Forward declarations
  class GenericMatrix;
  class GenericVector;
  class EpetraMatrix;
  class EpetraVector;

  /// This class implements the direct solution (LU factorization) for
  /// linear systems of the form Ax = b. It is a wrapper for the LU
  /// solver of Epetra.

  class EpetraLUSolver : public GenericLUSolver
  {
  public:

    /// Constructor
    EpetraLUSolver(std::string method="default");

    /// Constructor
    EpetraLUSolver(boost::shared_ptr<const GenericMatrix> A,
                   std::string method="default");

    /// Destructor
    ~EpetraLUSolver();

    /// Set operator (matrix)
    void set_operator(const boost::shared_ptr<const GenericMatrix> A);

    /// Get operator (matrix)
    const GenericMatrix& get_operator() const;

    /// Solve linear system Ax = b
    uint solve(GenericVector& x, const GenericVector& b);

    /// Solve linear system Ax = b
    uint solve(const GenericMatrix& A, GenericVector& x, const GenericVector& b);

    /// Solve linear system Ax = b
    uint solve(const EpetraMatrix& A, EpetraVector& x, const EpetraVector& b);

    /// Return a list of available solver methods
    static std::vector<std::pair<std::string, std::string> > methods();

    /// Default parameter values
    static Parameters default_parameters();

    /// Return informal string representation (pretty-print)
    std::string str(bool verbose) const;

  private:

    // Choose method / solver package
    std::string choose_method(std::string method) const;

    bool symbolic_factorized, numeric_factorized;

    // Operator (the matrix)
    boost::shared_ptr<const EpetraMatrix> A;

    // Epetra linear problem
    boost::scoped_ptr<Epetra_LinearProblem> linear_problem;

    // Linear solver
    boost::scoped_ptr<Amesos_BaseSolver> solver;

    // Solver method
    std::string method;

  };

}

#endif

#endif
