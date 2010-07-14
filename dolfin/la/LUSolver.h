// Copyright (C) 2007-2010 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Ola Skavhaug, 2008.
// Modified by Dag Lindbo, 2008.
// Modified by Anders Logg, 2008-2009.
// Modified by Kent-Andre Mardal, 2008.
//
// First added:  2007-07-03
// Last changed: 2010-07-11

#ifndef __LU_SOLVER_H
#define __LU_SOLVER_H

#include <string>
#include <boost/scoped_ptr.hpp>
#include "GenericLUSolver.h"

namespace dolfin
{

  // Forward declarations
  class GenericMatrix;
  class GenericVector;

  class LUSolver : public GenericLUSolver
  {

  /// LU solver for the built-in LA backends. The type can be "lu" or
  /// "cholesky". Cholesky is suitable only for symmetric positive-definite
  /// matrices. Cholesky is not yet suppprted for all backends (which will
  /// default to LU.

  public:

    /// Constructor
    LUSolver(std::string type = "lu");

    /// Constructor
    LUSolver(const GenericMatrix& A, std::string type = "lu");

    /// Destructor
    ~LUSolver();

    /// Set operator (matrix)
    void set_operator(const GenericMatrix& A);

    /// Solve linear system Ax = b
    uint solve(GenericVector& x, const GenericVector& b);

    /// Solve linear system
    uint solve(const GenericMatrix& A, GenericVector& x, const GenericVector& b);

    /// Default parameter values
    static Parameters default_parameters()
    {
      Parameters p("lu_solver");
      p.add("report", true);
      p.add("same_nonzero_pattern", false);
      p.add("reuse_factorization", false);
      return p;
    }

  private:

    // Solver
    boost::scoped_ptr<GenericLUSolver> solver;

  };
}

#endif
