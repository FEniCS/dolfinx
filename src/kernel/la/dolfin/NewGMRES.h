// Copyright (C) 2005 Johan Jansson.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg, 2005.

#ifndef __NEW_GMRES_H
#define __NEW_GMRES_H

#include <dolfin/constants.h>
#include <petsc/petscksp.h>

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

    /// Change rtol
    void changeRtol(real rt);
      
    /// Change abstol
    void changeAbstol(real at);
      
    /// Change dtol
    void changeDtol(real dt);
      
    /// Change maxits
    void changeMaxits(int mi);

    /// Return PETSc solver pointer
    KSP solver();
      
  private:

    // Tolerances:
    // rtol - the relative convergence tolerance (relative decrease in the residual norm)
    // abstol - the absolute convergence tolerance (absolute size of the residual norm)
    // dtol - the divergence tolerance (amount residual can increase before KSPDefaultConverged() 
    //	                 		concludes that the method is diverging)
    // maxits - maximum number of iterations to use

    real rtol;
    real abstol;
    real dtol;
    int maxits;

    // Maximum number of iterations
    unsigned int maxiter;

    // PETSc solver pointer
    KSP ksp;

  };

}

#endif
