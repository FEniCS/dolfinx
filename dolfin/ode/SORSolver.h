// Copyright (C) 2008 Benjamin Kehlet.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-10-11
// Last changed: 2008-10-11
//
// This class implements a simple SOR solver for systems of linear
// equations. The purpose is primarily to be able to solve systems
// with extended precision, as the other solvers make use of LA
// backends with are limited to double precision.

#ifndef __SOR_SOLVER_H
#define __SOR_SOLVER_H

#include <dolfin/common/types.h>
#include <dolfin/common/real.h>
#include <dolfin/la/uBLASDenseMatrix.h>

#define SOR_MAX_ITERATIONS 1000

namespace dolfin
{

  class SORSolver
  {
  public:

    // Will replace the initial guess in x
    static void SOR(uint n,
		    const real* A,
		    real* x,
		    const real* b,
		    const real& tol);

    // Solve Ax=b and precondition with matrix precond
    static void SOR_precond(uint n,
			    const real* A,
			    real* x,
			    const real* b,
			    const uBLASDenseMatrix& precond,
			    const real& tol );

    static void SOR_mat_with_preconditioning(uint n,
					     const real* A,
					     real* x,
					     const real* B,
					     const real& tol);

    static real err(uint n, const real* A, const real* x, const real* b);

  private:

    static void SOR_iteration(uint n,
			      const real* A,
			      const real* b,
			      real* x_new,
			      const real* x_prev);
  };

}

#endif
