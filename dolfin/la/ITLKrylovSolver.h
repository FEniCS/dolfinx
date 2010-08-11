// Copyright (C) 2008-2009 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Dag Lindbo, 2008.
// Modified by Anders Logg, 2008.
//
// First added:  2008-07-16
// Last changed: 2009-09-08

#ifdef HAS_MTL4

#ifndef __ITL_KRYLOV_SOLVER_H
#define __ITL_KRYLOV_SOLVER_H

#include <boost/shared_ptr.hpp>
#include <dolfin/common/types.h>
#include "GenericLinearSolver.h"

namespace dolfin
{

  /// Forward declarations
  class GenericMatrix;
  class GenericVector;
  class MTL4Matrix;
  class MTL4Vector;

  /// This class implements Krylov methods for linear systems
  /// of the form Ax = b. It is a wrapper for the Krylov solvers
  /// of ITL.

  class ITLKrylovSolver : public GenericLinearSolver
  {
  public:

    /// Create Krylov solver for a particular method and preconditioner
    ITLKrylovSolver(std::string method = "default", std::string pc_type = "default");

    /// Destructor
    ~ITLKrylovSolver();

    /// Set operator (matrix)
    void set_operator(const GenericMatrix& A);

    /// Set operator (matrix) and preconditioner matrix
    void set_operators(const GenericMatrix& A, const GenericMatrix& P);

    /// Solve linear system Ax = b and return number of iterations
    uint solve(GenericVector& x, const GenericVector& b);

    /// Solve linear system Ax = b and return number of iterations
    uint solve(MTL4Vector& x, const MTL4Vector& b);

    /// Solve linear system Ax = b and return number of iterations
    uint solve(const GenericMatrix& A, GenericVector& x, const GenericVector& b);

    /// Return informal string representation (pretty-print)
    std::string str(bool verbose) const;

    /// Default parameter values
    static Parameters default_parameters();

  private:

    /// Operator (the matrix)
    boost::shared_ptr<const MTL4Matrix> A;

    /// Matrix used to construct the preconditoner
    boost::shared_ptr<const MTL4Matrix> P;

    // Solver type
    std::string method;

    // Preconditioner type
    std::string pc_type;

  };

}

#endif

#endif
