// Copyright (C) 2008-2010 Kent-Andre Mardal and Garth N. Wells
// Licensed under the GNU LGPL Version 2.1.
//
// Last changed: 2010-07-16

#ifdef HAS_TRILINOS

#ifndef __EPETRA_LU_SOLVER_H
#define __EPETRA_LU_SOLVER_H

#include <boost/scoped_ptr.hpp>
#include "GenericLUSolver.h"

/// Forward declaration
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
    EpetraLUSolver();

    /// Constructor
    EpetraLUSolver(const GenericMatrix& A);

    /// Destructor
    ~EpetraLUSolver();

    /// Set operator (matrix)
    void set_operator(const GenericMatrix& A);

    /// Solve linear system Ax = b
    uint solve(GenericVector& x, const GenericVector& b);

    /// Solve linear system Ax = b
    uint solve(const GenericMatrix& A, GenericVector& x, const GenericVector& b);

    /// Solve linear system Ax = b
    uint solve(const EpetraMatrix& A, EpetraVector& x, const EpetraVector& b);

    /// Default parameter values
    static Parameters default_parameters();

    /// Return informal string representation (pretty-print)
    std::string str(bool verbose) const;

  private:

    boost::scoped_ptr<Epetra_LinearProblem> linear_problem;

  };

}

#endif





#endif


