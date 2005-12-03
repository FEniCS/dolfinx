// Copyright (C) 2004-2005 Johan Jansson.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg 2005.
// Modified by Johan Hoffman 2005.
// Modified by Andy R. Terrel 2005.
// Modified by Garth N. Wells 2005.
//
// First added:  2005-12-02
// Last changed:

#ifndef __KRYLOV_SOLVER_H
#define __KRYLOV_SOLVER_H

#include <petscksp.h>
#include <dolfin/constants.h>
#include <dolfin/Preconditioner.h>
#include <dolfin/LinearSolver.h>

namespace dolfin
{
  /// This class implements Krylov methods for linear systems
  /// of the form Ax = b. It is a wrapper for the GMRES solver
  /// of PETSc.
  
  class KrylovSolver : public LinearSolver
  {
  public:

    /// Create Krylov solver
    KrylovSolver();

    /// Create Krylov solver for a particular method
    KrylovSolver(KSPType type);

    /// Destructor
    ~KrylovSolver();

    /// Solve linear system Ax = b and return number of iterations
    uint solve(const Matrix& A, Vector& x, const Vector& b);
          
    /// Solve linear system Ax = b and return number of iterations
    uint solve(const VirtualMatrix& A, Vector& x, const Vector& b);
    
    /// FIXME: Options below should be moved to some parameter system,
    /// FIXME: not very nice to have a long list of setFoo() functions.

    /// Set Krylov method type
    void setType(KSPType type);

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

    // True if we should report the number of iterations
    bool report;

    // PETSc solver pointer
    KSP ksp;

    // Type of Krylov method
    KSPType ksptype;

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
