// Copyright (C) 2011 Benjamin Kehlet
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2011-02-16
// Last changed: 2011-02-16

#ifndef __HIGH_PRECISION_H
#define __HIGH_PRECISION_H


#include <dolfin/common/types.h>
#include <dolfin/common/real.h>
#include <dolfin/la/uBLASDenseMatrix.h>

#define MAX_ITERATIONS 1000

namespace dolfin
{

  class HighPrecision
  {
  public:

    static void real_mat_exp(uint n, real* E, const real* A, const uint p);

    /// Matrix multiplication res = A*B
    static void real_mat_prod(uint n, real* res, const real* A, const real* B);

    /// Matrix multiplication A = A * B
    static void real_mat_prod_inplace(uint n, real* A, const real* B);

    /// Matrix vector product y = Ax
    static void real_mat_vector_prod(uint n, real* y, const real* A, const real* x);

    /// Matrix power A = B^q
    static void real_mat_pow(uint n, real* A, const real* B, uint q);

    /// Solve AX = B by first inverting A in double precision, then
    /// doing Gauss-Seidel iteration with A inverted as preconditioner.
    /// Will replace the initial guess in x
    static void real_solve_mat_with_preconditioning(uint n,
                                      const real* A,
                                      real* x,
                                      const real* B,
                                      const real& tol);


    /// Solves Ax = b by Gauss-Seidel iterations
    static void real_solve(uint n, const real* A, real* x, const real* b, const real& tol);

    /// Solve Ax = b with preconditioner
    static void real_solve_precond(uint n,
                       const real* A,
                       real* x,
                       const real* b,
                       const uBLASDenseMatrix& precond,
                       const real& tol );
  };
}
#endif
