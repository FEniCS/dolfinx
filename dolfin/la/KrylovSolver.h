// Copyright (C) 2007-2009 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Ola Skavhaug, 2008.
// Modified by Anders Logg, 2008.
//
// First added:  2007-07-03
// Last changed: 2010-04-22

#ifndef __KRYLOV_SOLVER_H
#define __KRYLOV_SOLVER_H

#include "GenericLinearSolver.h"

namespace dolfin
{

  class GenericMatrix;
  class GenericVector;
  class Parameters;
  class uBLASKrylovSolver;
  class PETScKrylovSolver;
  class EpetraKrylovSolver;
  class ITLKrylovSolver;

  /// This class defines an interface for a Krylov solver. The approproiate solver
  /// is chosen on the basis of the matrix/vector type.

  class KrylovSolver : public GenericLinearSolver
  {
  public:

    /// Create Krylov solver
    KrylovSolver(std::string solver_type = "default",
                 std::string pc_type = "default");

    /// Destructor
    ~KrylovSolver();

    /// Solve linear system Ax = b
    uint solve(const GenericMatrix& A, GenericVector& x, const GenericVector& b);

    /// Default parameter values
    static Parameters default_parameters();

  private:

    // Krylov method
    std::string solver_type;

    // Preconditioner type
    std::string pc_type;

    // Solvers
    uBLASKrylovSolver* ublas_solver;
#ifdef HAS_PETSC
    PETScKrylovSolver* petsc_solver;
#else
    int* petsc_solver;
#endif
#ifdef HAS_TRILINOS
    EpetraKrylovSolver* epetra_solver;
#else
    int* epetra_solver;
#endif
#ifdef HAS_MTL4
    ITLKrylovSolver* itl_solver;
#else
    int* itl_solver;
#endif

  };
}

#endif
