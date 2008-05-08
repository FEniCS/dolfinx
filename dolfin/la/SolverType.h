// Copyright (C) 2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-05-08
// Last changed: 2008-05-08

#ifndef __SOLVER_TYPE_H
#define __SOLVER_TYPE_H

namespace dolfin
{

  /// List of predefined solvers

  enum SolverType
  {
    lu,            // LU factorization
    cg,            // Krylov conjugate gradient method
    gmres,         // Krylov GMRES method
    bicgstab,      // Krylov stabilised biconjugate gradient squared method 
    default_solver // Default Krylov solver
  };

}

#endif
