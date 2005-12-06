// Copyright (C) 2004-2005 Johan Jansson.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg 2005.
// Modified by Johan Hoffman 2005.
// Modified by Andy R. Terrel 2005.
// Modified by Garth N. Wells 2005.
//
// First added:  2005-12-02
// Last changed: 2005-12-06

#ifndef __KRYLOV_SOLVER_H
#define __KRYLOV_SOLVER_H

#include <petscksp.h>
#include <dolfin/constants.h>
#include <dolfin/Preconditioner.h>
#include <dolfin/LinearSolver.h>

namespace dolfin
{
  /// This class implements Krylov methods for linear systems
  /// of the form Ax = b. It is a wrapper for the Krylov solvers
  /// of PETSc.
  
  class KrylovSolver : public LinearSolver
  {
  public:

    // Krylov methods
    enum SolverType { 
        bicgstab,       // Stabilised biconjugate gradient squared method 
        cg,             // Conjugate gradient method
        default_solver, // Default PETSc solver (use when setting solver from command line)
        gmres           // GMRES method
     };

    // Preconditioners
    enum PreconditionerType { 
        default_pc,   // Deafault PETSc preconditioner (use when setting solver from command line)
        icc,          // Incomplete Cholesky  
        ilu,          // Incomplete LU
        jacobi,       // Jacobi
        none,         // No preconditioning
        sor           // SOR (successive over relaxation)
       };

    /// Create Krylov solver (GMRES with incomplete LU preconditioning)
    KrylovSolver();

    /// Create Krylov solver for a particular method (with incomplete LU preconditioning)
    KrylovSolver(SolverType solvertype);

    /// Create Krylov solver for a particular method and preconditioner
    KrylovSolver(SolverType solvertype, PreconditionerType preconditionertype);

    /// Destructor
    ~KrylovSolver();

    /// Solve linear system Ax = b and return number of iterations
    uint solve(const Matrix& A, Vector& x, const Vector& b);
          
    /// Solve linear system Ax = b and return number of iterations
    uint solve(const VirtualMatrix& A, Vector& x, const Vector& b);
    
    /// FIXME: Options below should be moved to some parameter system,
    /// FIXME: not very nice to have a long list of setFoo() functions.

    /// Set Krylov method type
    void setType(const SolverType solvertype);

    /// Set PETSc preconditioner 
    void setPreconditioner(const PreconditionerType preconditionertype);

    /// Change whether solver should report the number iterations
    void setReport(bool report);

    /// Change rtol
    void setRtol(real rtol);
      
    /// Change abstol
    void setAtol(real atol);
      
    /// Change dtol
    void setDtol(real dtol);
      
    /// Change maxiter
    void setMaxiter(int maxiter);

    /// Set preconditioner
    void setPreconditioner(Preconditioner &pc);

    /// Return PETSc solver pointer
    KSP solver(){
      return ksp;
    }

    /// Display solver data
    void disp() const;
     
  private:

    // Initialize KSP solver
    void init(uint M, uint N);

    // Create preconditioner matrix for virtual matrix
    void createVirtualPreconditioner(const VirtualMatrix& A);

    // Get PETSc method identifier 
    KSPType getKSPType(const SolverType type) const;

    // Get PETSc preconditioner identifier
    PCType getPCType(const PreconditionerType type) const;

    // True if we should report the number of iterations
    bool report;

    // Solver type
    SolverType _solvertype;
    
    // Solver type
    PreconditionerType _preconditionertype;

    // PETSc solver pointer
    KSP ksp;

    // Diagonal matrix used for preconditioning with virtual matrix
    Mat B;

    // Size of old system (need to reinitialize when changing)
    uint M;
    uint N;

    // Optional DOLFIN preconditioner (can't be set with PCSetType yet)
    Preconditioner *dolfin_pc;

  };

}

#endif
