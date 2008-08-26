// Copyright (C) 2004-2005 Johan Jansson.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2005-2008.
// Modified by Johan Hoffman, 2005.
// Modified by Andy R. Terrel, 2005.
// Modified by Garth N. Wells, 2005-2007.
//
// First added:  2005-12-02
// Last changed: 2008-08-25

#ifndef __PETSC_KRYLOV_SOLVER_H
#define __PETSC_KRYLOV_SOLVER_H

#ifdef HAS_PETSC

#include <dolfin/common/types.h>
#include "GenericLinearSolver.h"
#include "enums_la.h"
#include "PETScPreconditioner.h"

namespace dolfin
{

  /// Forward declarations
  class GenericMatrix;
  class GenericVector;
  class PETScMatrix;
  class PETScVector;
  class PETScKrylovMatrix;

  /// This class implements Krylov methods for linear systems
  /// of the form Ax = b. It is a wrapper for the Krylov solvers
  /// of PETSc.
  
  class PETScKrylovSolver : public GenericLinearSolver
  {
  public:

    /// Create Krylov solver for a particular method and preconditioner
    PETScKrylovSolver(dolfin::SolverType method=default_solver,
                      dolfin::PreconditionerType pc=default_pc);

    /// Create Krylov solver for a particular method and PETScPreconditioner
    PETScKrylovSolver(dolfin::SolverType method, PETScPreconditioner& PETScPreconditioner);

    /// Destructor
    ~PETScKrylovSolver();

    /// Solve linear system Ax = b and return number of iterations
    uint solve(const GenericMatrix& A, GenericVector& x, const GenericVector& b);

    /// Solve linear system Ax = b and return number of iterations
    uint solve(const PETScMatrix& A, PETScVector& x, const PETScVector& b);
          
    /// Solve linear system Ax = b and return number of iterations
    uint solve(const PETScKrylovMatrix& A, PETScVector& x, const PETScVector& b);
    
    /// Display solver data
    void disp() const;
     
  private:

    /// Initialize KSP solver
    void init(uint M, uint N);

    /// Read parameters from database
    void readParameters();
    
    /// Set solver
    void setSolver();

    /// Set PETScPreconditioner
    void setPETScPreconditioner();
    
    /// Report the number of iterations
    void writeReport(int num_iterations);

    /// Get PETSc method identifier 
    KSPType getType(dolfin::SolverType method) const;

    /// Krylov method
    SolverType method;

    /// PETSc PETScPreconditioner
    PreconditionerType pc_petsc;

    /// DOLFIN PETScPreconditioner
    PETScPreconditioner* pc_dolfin;

    /// PETSc solver pointer
    KSP ksp;

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
