// Copyright (C) 2008 Kent-Andre Mardal.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2008.
// Modified by Garth N. Wells, 2009.
//
// Last changed: 2009-09-08

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
                       std::string pc_type = "default");

    /// Create Krylov solver for a particular method and TrilinosPreconditioner
    EpetraKrylovSolver(std::string method, TrilinosPreconditioner& preconditioner);

    /// Destructor
    ~EpetraKrylovSolver();

    /// Solve linear system Ax = b and return number of iterations
    uint solve(const GenericMatrix& A, GenericVector& x, const GenericVector& b);

    /// Solve linear system Ax = b and return number of iterations
    uint solve(const EpetraMatrix& A, EpetraVector& x, const EpetraVector& b);

    /// Default parameter values
    static Parameters default_parameters();

    /// Return informal string representation (pretty-print)
    std::string str(bool verbose) const;

    /// Return pointer to Aztec00
    boost::shared_ptr<AztecOO> aztecoo() const;

  private:

    // Solver type
    std::string method;

    // Available solvers and preconditioners
    static const std::map<std::string, int> methods;
    static const std::map<std::string, int> pc_methods;

    // Underlying solver
    boost::shared_ptr<AztecOO> solver;

    // Preconditioner
    boost::shared_ptr<TrilinosPreconditioner> preconditioner;

    bool preconditioner_set;

  };

}

#endif

#endif
