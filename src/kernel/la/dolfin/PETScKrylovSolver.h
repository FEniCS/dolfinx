// Copyright (C) 2004-2005 Johan Jansson.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg 2005-2006.
// Modified by Johan Hoffman 2005.
// Modified by Andy R. Terrel 2005.
// Modified by Garth N. Wells 2005-2007.
//
// First added:  2005-12-02
// Last changed: 2007-07-31

#ifndef __PETSC_KRYLOV_SOLVER_H
#define __PETSC_KRYLOV_SOLVER_H

#ifdef HAVE_PETSC_H

#include <dolfin/PETScLinearSolver.h>
#include <dolfin/constants.h>
#include <dolfin/Parametrized.h>
#include <dolfin/Preconditioner.h>
#include <dolfin/KrylovMethod.h>
#include <dolfin/PETScPreconditioner.h>

namespace dolfin
{

  /// Forward declarations
  class PETScMatrix;
  class PETScVector;
  class PETScKrylovMatrix;

  /// This class implements Krylov methods for linear systems
  /// of the form Ax = b. It is a wrapper for the Krylov solvers
  /// of PETSc.
  
  class PETScKrylovSolver : public PETScLinearSolver, public Parametrized
  {
  public:

    /// Create Krylov solver for a particular method and preconditioner
    PETScKrylovSolver(KrylovMethod method = default_method, Preconditioner pc = default_pc);

    /// Create Krylov solver for a particular method and PETScPreconditioner
    PETScKrylovSolver(KrylovMethod method, PETScPreconditioner& PETScPreconditioner);

    /// Destructor
    ~PETScKrylovSolver();

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
    KSPType getType(KrylovMethod method) const;

    /// Krylov method
    KrylovMethod method;

    /// PETSc PETScPreconditioner
    Preconditioner pc_petsc;

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
