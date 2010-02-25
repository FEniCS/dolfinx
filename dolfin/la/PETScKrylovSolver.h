// Copyright (C) 2004-2005 Johan Jansson.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2005-2009.
// Modified by Johan Hoffman, 2005.
// Modified by Andy R. Terrel, 2005.
// Modified by Garth N. Wells, 2005-2010.
//
// First added:  2005-12-02
// Last changed: 2010-02-25

#ifndef __DOLFIN_PETSC_KRYLOV_SOLVER_H
#define __DOLFIN_PETSC_KRYLOV_SOLVER_H

#ifdef HAS_PETSC

#include <map>
#include <petscksp.h>
#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include <dolfin/common/types.h>
#include "GenericLinearSolver.h"

namespace dolfin
{

  /// Forward declarations
  class GenericMatrix;
  class GenericVector;
  class PETScMatrix;
  class PETScVector;
  class PETScKrylovMatrix;
  class PETScPreconditioner;
  class PETScUserPreconditioner;

  /// This class implements Krylov methods for linear systems
  /// of the form Ax = b. It is a wrapper for the Krylov solvers
  /// of PETSc.

  class PETScKrylovSolver : public GenericLinearSolver
  {
  public:

    /// Create Krylov solver for a particular method and preconditioner
    PETScKrylovSolver(std::string method = "default", std::string pc_type = "default");

    /// Create Krylov solver for a particular method and PETScPreconditioner
    PETScKrylovSolver(std::string method, PETScUserPreconditioner& PETScUserPreconditioner);

    /// Create solver from given PETSc KSP pointer
    explicit PETScKrylovSolver(boost::shared_ptr<KSP> ksp);

    /// Destructor
    ~PETScKrylovSolver();

    /// Solve linear system Ax = b and return number of iterations
    uint solve(const GenericMatrix& A, GenericVector& x, const GenericVector& b);

    /// Solve linear system Ax = b and return number of iterations
    uint solve(const PETScMatrix& A, PETScVector& x, const PETScVector& b);

    /// Solve linear system Ax = b and return number of iterations
    uint solve(const PETScKrylovMatrix& A, PETScVector& x, const PETScVector& b);

    /// Return informal string representation (pretty-print)
    std::string str(bool verbose) const;

    /// Default parameter values
    static Parameters default_parameters();

    /// Return PETSc KSP pointer
    boost::shared_ptr<KSP> ksp() const;

  private:

    /// Initialize KSP solver
    void init(uint M, uint N);

    /// Report the number of iterations
    void write_report(int num_iterations);

    /// Krylov method
    std::string method;

    // Available solvers and preconditioners
    static const std::map<std::string, const KSPType> methods;
    static const std::map<std::string, const PCType> pc_methods;

    /// DOLFIN-defined PETScUserPreconditioner
    PETScUserPreconditioner* pc_dolfin;

    /// PETSc solver pointer
    boost::shared_ptr<KSP> _ksp;

    /// Preconditioner
    boost::scoped_ptr<PETScPreconditioner> preconditioner;

    /// True if we have read parameters
    bool parameters_read;

  };

}

#endif

#endif
