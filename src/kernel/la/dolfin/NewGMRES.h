// Copyright (C) 2005 Johan Jansson.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg, 2005.
// Modified by Johan Hoffman, 2005.

#ifndef __NEW_GMRES_H
#define __NEW_GMRES_H

#include <dolfin/constants.h>
#include <petsc/petscksp.h>
#include <dolfin/NewPreconditioner.h>

namespace dolfin
{
  
  class NewMatrix;
  class NewVector;
  class VirtualMatrix;

  /// This is just a template. Write documentation here.
  
  class NewGMRES
  {
  public:

    /// Create GMRES solver
    NewGMRES();

    /// Destructor
    ~NewGMRES();

    /// Solve linear system Ax = b for a given right-hand side b
    void solve(const NewMatrix& A, NewVector& x, const NewVector& b);

    /// Solve linear system Ax = b for a given right-hand side b
    void solve(const VirtualMatrix& A, NewVector& x, const NewVector& b);

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
    void setPreconditioner(NewPreconditioner &pc);

    /// Return PETSc solver pointer
    KSP solver();

    /// Display GMRES solver data
    void disp() const;
     
  private:

    // True if we should report the number of iterations
    bool report;

    // PETSc solver pointer
    KSP ksp;

  };

}

#endif
