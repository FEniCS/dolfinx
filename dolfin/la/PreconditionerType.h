// Copyright (C) 2006-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-08-15
// Last changed: 2008-05-08

#ifndef __PRECONDITIONER_TYPE_H
#define __PRECONDITIONER_TYPE_H

namespace dolfin
{

  /// List of predefined preconditioners

  enum PreconditionerType
  {
    none,      // No preconditioning
    jacobi,    // Jacobi
    sor,       // SOR (successive over relaxation)
    ilu,       // Incomplete LU factorization
    icc,       // Incomplete Cholesky factorization
    amg,       // Algebraic multigrid (through Hypre when available)
    default_pc // Default choice of preconditioner
  };

}

#endif
