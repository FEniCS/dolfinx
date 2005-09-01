// Copyright (C) 2004-2005 Johan Jansson.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg 2005.
// Modified by Johan Hoffman 2005.
// Modified by Garth N. Wells 2005.
//
// First added:  2004-06-22
// Last changed: 2005-09-01

#ifndef __GMRES_H
#define __GMRES_H

#include <petscksp.h>
#include <dolfin/constants.h>
#include <dolfin/Preconditioner.h>
#include <dolfin/LinearSolver.h>

namespace dolfin
{

  /// This class implements the GMRES method for linear systems
  /// of the form Ax = b. It is a wrapper for the GMRES solver
  /// of PETSc.
  
  class GMRES : public LinearSolver
  {
  public:

    /// Create GMRES solver
    GMRES();

    /// Destructor
    ~GMRES();

    /// Solve linear system Ax = b
    void solve(const Matrix& A, Vector& x, const Vector& b);
          
    /// Solve linear system Ax = b (matrix-free version)
    void solve(const VirtualMatrix& A, Vector& x, const Vector& b);
    
    /// FIXME: Options below should be moved to some parameter system,
    /// FIXME: not very nice to have a long list of setFoo() functions.

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
    KSP solver();

    /// Display GMRES solver data
    void disp() const;
     
  private:

    // Create preconditioner matrix for virtual matrix
    void createVirtualPreconditioner(const VirtualMatrix& A);

    // True if we should report the number of iterations
    bool report;

    // PETSc solver pointer
    KSP ksp;

    // Diagonal matrix used for preconditioning with virtual matrix
    Mat B;

  };

}

#endif
