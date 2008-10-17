// Copyright (C) 2008 Benjamin Kehlet
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-10-11
// Last changed: 2008-10-11
//
// This class implements a simple SOR solver for systems of linear
// equation. The purpose is primarily to be able to solve systems
// with extended precision, as the other solvers make use of LA backends
// with are limited to double precision

#ifndef __SOR_SOLVER_H
#define __SOR_SOLVER_H

#include <dolfin/common/types.h>
#include <dolfin/log/log.h>
#include <dolfin/common/real.h>
#include <iostream>
#include <dolfin/la/uBLASDenseMatrix.h>

#define SOR_MAX_ITERATIONS 1000

namespace dolfin
{

  class SORSolver {
   public:

    // Will replace the initial guess in x
    static void SOR(uint n, const real* A, 
		    real* x, const real* b, 
		    const real& epsilon); 

    // Compute A_inv*A and Ainv*b
    static void precondition(uint n, 
			     const uBLASDenseMatrix& Ainv, 
			     real* A, real* b,
			     real* Ainv_A, real* Ainv_b);

    static void printMatrix(const uint n, const real* A);
    static void printVector(const uint n, const real* x);

  private:

    static void _SOR_iteration(uint n, 
			       const real* A, 
			       const real* b, 
			       real* x_new, 
			       const real* x_prev);

  };
}

#endif
