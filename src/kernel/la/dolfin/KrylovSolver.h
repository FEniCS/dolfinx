// Copyright (C) 2004-2005 Johan Jansson.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg 2005-2006.
// Modified by Johan Hoffman 2005.
// Modified by Andy R. Terrel 2005.
// Modified by Garth N. Wells 2005.
//
// First added:  2005-12-02
// Last changed: 2006-03-14

#ifndef __KRYLOV_SOLVER_H
#define __KRYLOV_SOLVER_H

#include <petscksp.h>
#include <dolfin/constants.h>
#include <dolfin/Parametrized.h>
#include <dolfin/Preconditioner.h>
#include <dolfin/LinearSolver.h>

namespace dolfin
{
  /// This class implements Krylov methods for linear systems
  /// of the form Ax = b. It is a wrapper for the Krylov solvers
  /// of PETSc.
  
  class KrylovSolver : public LinearSolver, public Parametrized
  {
  public:

    /// Krylov methods
    enum Type
    { 
      bicgstab,       // Stabilised biconjugate gradient squared method 
      cg,             // Conjugate gradient method
      default_solver, // Default PETSc solver (use when setting solver from command line)
      gmres           // GMRES method
    };

    /// Create Krylov solver with PETSc default method and preconditioner
    KrylovSolver();

    /// Create Krylov solver for a particular method with default PETSc preconditioner
    KrylovSolver(Type solver);

    /// Create Krylov solver with default PETSc method and a particular preconditioner
    KrylovSolver(Preconditioner::Type preconditioner);

    /// Create Krylov solver with default PETSc method and a particular preconditioner
    KrylovSolver(Preconditioner& preconditioner);

    /// Create Krylov solver for a particular method and preconditioner
    KrylovSolver(Type solver, Preconditioner::Type preconditioner);

    /// Create Krylov solver for a particular method and preconditioner
    KrylovSolver(Type solver, Preconditioner& preconditioner);

    /// Destructor
    ~KrylovSolver();

    /// Solve linear system Ax = b and return number of iterations
    uint solve(const Matrix& A, Vector& x, const Vector& b);
          
    /// Solve linear system Ax = b and return number of iterations
    uint solve(const VirtualMatrix& A, Vector& x, const Vector& b);
    
    /// Return PETSc solver pointer
    KSP solver();

    /// Display solver data
    void disp() const;
     
  private:

    /// Initialize KSP solver
    void init(uint M, uint N);

    /// Read parameters from database
    void readParameters();
    
    /// Set solver
    void setSolver();

    /// Set preconditioner
    void setPreconditioner();
    
    /// Report the number of iterations
    void writeReport(int num_iterations);

    /// Get PETSc method identifier 
    KSPType getType(const Type type) const;

    /// PETSc solver type
    Type type;

    /// PETSc preconditioner
    Preconditioner::Type pc_petsc;

    /// DOLFIN preconditioner
    Preconditioner* pc_dolfin;

    /// PETSc solver pointer
    KSP ksp;

    /// Size of old system (need to reinitialize when changing)
    uint M;
    uint N;
    
  };

}

#endif
