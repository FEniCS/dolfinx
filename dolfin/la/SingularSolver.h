// Copyright (C) 2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2005-09-19
// Last changed: 2008-08-25

#ifndef __SINGULAR_SOLVER_H
#define __SINGULAR_SOLVER_H

#include <dolfin/common/types.h>
#include <dolfin/parameter/Parametrized.h>
#include "LinearSolver.h"

namespace dolfin
{
  /// This class provides a linear solver for singular linear systems
  /// Ax = b where A has a one-dimensional null-space (kernel). This
  /// may happen for example when solving Poisson's equation with
  /// pure Neumann boundary conditions.
  ///
  /// The solver attempts to create an extended non-singular system
  /// by adding the constraint [1, 1, 1, ...]^T x = 0.
  ///
  /// If an optional mass matrix M is supplied, the solver attempts
  /// to create an extended non-singular system by adding the
  /// constraint m^T x = 0 where m is the lumped mass matrix. This
  /// corresponds to setting the average (integral) of the finite
  /// element function with coefficients x to zero.
  ///
  /// The solver makes not attempt to check that the null-space is
  /// indeed one-dimensional. It is also assumed that the system
  /// Ax = b retains its sparsity pattern between calls to solve().
  
  class SingularSolver : public Parametrized
  {
  public:

    /// Create linear solver
    SingularSolver(dolfin::SolverType solver_type=lu,
                   dolfin::PreconditionerType pc_type=ilu);

    /// Destructor
    ~SingularSolver();

    /// Solve linear system Ax = b
    uint solve(const GenericMatrix& A, GenericVector& x, const GenericVector& b);

    /// Solve linear system Ax = b using mass matrix M for setting constraint
    uint solve(const GenericMatrix& A, GenericVector& x, const GenericVector& b,
               const GenericMatrix& M);

  private:

    // Initialize extended system
    void init(const GenericMatrix& A);

    // Create extended system
    void create(const GenericMatrix& A, const GenericVector& b, const GenericMatrix* M);

    // Linear solver
    LinearSolver linear_solver;

    // Extended matrix
    GenericMatrix* B;

    // Solution of extended system
    GenericVector* y;
    
    // Extended vector
    GenericVector* c;

  };

}

#endif
