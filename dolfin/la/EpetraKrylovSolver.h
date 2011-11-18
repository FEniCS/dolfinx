// Copyright (C) 2008-2011 Kent-Andre Mardal and Garth N. Wells
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
// Modified by Anders Logg 2008-2011
// Modified by Garth N. Wells 2009
//
// First added:  2008
// Last changed: 2011-10-19

#ifdef HAS_TRILINOS

#ifndef __EPETRA_KRYLOV_SOLVER_H
#define __EPETRA_KRYLOV_SOLVER_H

#include <map>
#include <string>
#include <boost/shared_ptr.hpp>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/common/types.h>
#include "GenericLinearSolver.h"

// Forward declarations
class AztecOO;

namespace dolfin
{

  // Forward declarations
  class GenericMatrix;
  class GenericVector;
  class EpetraMatrix;
  class EpetraVector;
  class EpetraKrylovMatrix;
  class EpetraUserPreconditioner;
  class TrilinosPreconditioner;

  /// This class implements Krylov methods for linear systems
  /// of the form Ax = b. It is a wrapper for the Krylov solvers
  /// of Epetra.

  class EpetraKrylovSolver : public GenericLinearSolver
  {
  public:

    /// Create Krylov solver for a particular method and preconditioner
    EpetraKrylovSolver(std::string method = "default",
                       std::string preconditioner = "default");

    /// Create Krylov solver for a particular method and TrilinosPreconditioner
    EpetraKrylovSolver(std::string method, TrilinosPreconditioner& preconditioner);

    /// Destructor
    ~EpetraKrylovSolver();

    /// Set the operator (matrix)
    void set_operator(const boost::shared_ptr<const GenericMatrix> A);

    /// Set the operator (matrix)
    void set_operators(const boost::shared_ptr<const GenericMatrix> A,
                       const boost::shared_ptr<const GenericMatrix> P);

    /// Get the operator (matrix)
    const GenericMatrix& get_operator() const;

    /// Solve linear system Ax = b and return number of iterations
    uint solve(GenericVector& x, const GenericVector& b);

    /// Solve linear system Ax = b and return number of iterations
    uint solve(EpetraVector& x, const EpetraVector& b);

    /// Solve linear system Ax = b and return number of iterations
    uint solve(const GenericMatrix& A, GenericVector& x, const GenericVector& b);

    /// Solve linear system Ax = b and return number of iterations
    uint solve(const EpetraMatrix& A, EpetraVector& x, const EpetraVector& b);

    /// Return a list of available solver methods
    static std::vector<std::pair<std::string, std::string> > methods();

    /// Return a list of available preconditioners
    static std::vector<std::pair<std::string, std::string> > preconditioners();

    /// Default parameter values
    static Parameters default_parameters();

    /// Return informal string representation (pretty-print)
    std::string str(bool verbose) const;

    /// Return pointer to Aztec00
    boost::shared_ptr<AztecOO> aztecoo() const;

  private:

    // Solver type
    std::string method;

    // Available solvers
    static const std::map<std::string, int> _methods;

    // Underlying solver
    boost::shared_ptr<AztecOO> solver;

    // Preconditioner
    boost::shared_ptr<TrilinosPreconditioner> preconditioner;

    // Operator (the matrix)
    boost::shared_ptr<const EpetraMatrix> A;

    // Matrix used to construct the preconditoner
    boost::shared_ptr<const EpetraMatrix> P;

    bool preconditioner_set;

  };

}

#endif

#endif
