// Copyright (C) 2007-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Ola Skavhaug 2008.
//
// First added:  2007-04-30
// Last changed: 2008-05-23

#ifndef __SOLVE_H
#define __SOLVE_H

#include <dolfin/common/types.h>
#include "SolverType.h"
#include "PreconditionerType.h"
#include "NormalizationType.h"

namespace dolfin
{  

  class GenericMatrix;
  class GenericVector;

  /// Solve linear system Ax = b
  void solve(const GenericMatrix& A, GenericVector& x, const GenericVector& b,
             SolverType solver_type=lu, PreconditionerType pc_type=ilu);

  /// Compute residual ||Ax - b||
  real residual(const GenericMatrix& A, const GenericVector& x, const GenericVector& b);

  /// Normalize vector according to given normalization type
  real normalize(GenericVector& x, NormalizationType normalization_type=norm);
  
  /// Solve linear system Ax = b
  //void solve(const PETScKrylovMatrix& A, PETScVector& x, const PETScVector& b);
  
}

#endif
