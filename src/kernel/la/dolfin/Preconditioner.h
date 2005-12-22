// Copyright (C) 2005 Johan Jansson.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg 2005.
// Modified Garth N. Wells 2005
//
// First added:  2005
// Last changed: 2005-12-09

#ifndef __PRECONDITIONER_H
#define __PRECONDITIONER_H

#include <petscksp.h>
#include <petscpc.h>

namespace dolfin
{

  class Vector;

  /// This class specifies the interface for user-defined Krylov
  /// method preconditioners. A user wishing to implement her own
  /// preconditioner needs only supply a function that approximately
  /// solves the linear system given a right-hand side.

  class Preconditioner
  {
  public:

    // PETSC preconditioners
    enum Type
    { 
      default_pc, // Default PETSc preconditioner (use when setting solver from command line)
      hypre_amg,  // Hypre algerbraic multigrid
      icc,        // Incomplete Cholesky
      ilu,        // Incomplete LU
      jacobi,     // Jacobi
      sor,        // SOR (successive over relaxation)
      none        // No preconditioning
    };

    /// Constructor
    Preconditioner();

    /// Destructor
    virtual ~Preconditioner();

    static void setup(const KSP ksp, Preconditioner &pc);

    /// Solve linear system approximately for given right-hand side b
    virtual void solve(Vector& x, const Vector& b) = 0;

    /// Friends
    friend class KrylovSolver;

  protected:

    PC petscpc;

  private:

    static int PCApply(PC pc, Vec x, Vec y);
    static int PCCreate(PC pc);

    /// Return PETSc preconditioner type
    static PCType getType(const Type type);

  };

}

#endif
