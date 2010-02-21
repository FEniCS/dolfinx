// Copyright (C) 2004-2005 Johan Jansson.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2005-2009.
// Modified by Johan Hoffman, 2005.
// Modified by Andy R. Terrel, 2005.
// Modified by Garth N. Wells, 2005-2009.
//
// First added:  2005-12-02
// Last changed: 2009-09-08

#ifndef __PETSC_KRYLOV_SOLVER_H
#define __PETSC_KRYLOV_SOLVER_H

#ifdef HAS_PETSC

#include <map>
#include <petscksp.h>
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

  private:

    /// Initialize KSP solver
    void init(uint M, uint N);

    /// Read parameters from database
    void read_parameters();

    /// Set PETScPreconditioner
    void set_petsc_preconditioner();

    /// Report the number of iterations
    void write_report(int num_iterations);

    /// Krylov method
    std::string method;

    /// PETSc preconditioner type
    std::string pc_petsc;

    // Available solvers and preconditioners
    static const std::map<std::string, const KSPType> methods;
    static const std::map<std::string, const PCType> pc_methods;

    /// DOLFIN-defined PETScUserPreconditioner
    PETScUserPreconditioner* pc_dolfin;

    /// PETSc solver pointer
    boost::shared_ptr<KSP> ksp;

    /// Size of old system (need to reinitialize when changing)
    uint M;
    uint N;

    /// True if we have read parameters
    bool parameters_read;

    // FIXME: Required to avoid PETSc bug with Hypre. See explanation inside
    //        PETScKrylovSolver:init(). Can be removed when PETSc is patched.
    bool pc_set;
  };

}

#endif

#endif
