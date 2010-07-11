// Copyright (C) 2005-2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2009-2010.
//
// First added:  2005
// Last changed: 2010-02-15

#ifndef __DOLFIN_PETSC_LU_SOLVER_H
#define __DOLFIN_PETSC_LU_SOLVER_H

#ifdef HAS_PETSC

#include <map>
#include <boost/shared_ptr.hpp>
#include <petscksp.h>
#include <petscpc.h>
#include "GenericLUSolver.h"
#include "PETScObject.h"

namespace dolfin
{
  /// Forward declarations
  class GenericMatrix;
  class GenericVector;
  class PETScKrylovMatrix;
  class PETScMatrix;
  class PETScVector;

  /// This class implements the direct solution (LU factorization) for
  /// linear systems of the form Ax = b. It is a wrapper for the LU
  /// solver of PETSc.

  class PETScLUSolver : public GenericLUSolver, public PETScObject
  {
  public:

    /// Constructor
    PETScLUSolver(std::string lu_package="default");

    /// Constructor
    PETScLUSolver(const GenericMatrix& A, std::string lu_package="default");

    /// Constructor
    PETScLUSolver(boost::shared_ptr<const GenericMatrix> A,
                  std::string lu_package="default");

    /// Destructor
    ~PETScLUSolver();

    /// Set operator (matrix)
    void set_operator(const GenericMatrix& A);

    /// Solve linear system Ax = b
    uint solve(GenericVector& x, const GenericVector& b);

    /// LU-factor the sparse matrix A if UMFPACK is installed
    void factorize();

    /// Solve linear system Ax = b
    uint solve(const GenericMatrix& A, GenericVector& x, const GenericVector& b);

    /// Solve linear system Ax = b
    uint solve(const PETScMatrix& A, PETScVector& x, const PETScVector& b);

    /// Solve factorized system
    uint solve_factorized(GenericVector& x, const GenericVector& b) const;

    /// Return informal string representation (pretty-print)
    std::string str(bool verbose) const;

    /// Default parameter values
    static Parameters default_parameters();

  private:

    /// LU method
    std::string lu_package;

    // Available LU solvers
    static const std::map<std::string, const MatSolverPackage> lu_packages;

    // Select LU solver type
    void select_solver();

    // Initialise solver
    void init_solver();

    // Pritn pre-solve report
    void pre_report(const PETScMatrix& A) const;

    /// PETSc solver pointer
    boost::shared_ptr<KSP> ksp;

    // Operator (the matrix)
    boost::shared_ptr<const GenericMatrix> A;

  };

}

#endif

#endif
