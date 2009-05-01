// Copyright (C) 2007-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Ola Skavhaug 2008.
//
// First added:  2007-04-30
// Last changed: 2008-08-25

#ifndef __SOLVE_H
#define __SOLVE_H

#include <dolfin/common/types.h>
#include "enums_la.h"

namespace dolfin
{

  class GenericMatrix;
  class GenericVector;

  /// Solve linear system Ax = b
  void solve(const GenericMatrix& A, GenericVector& x, const GenericVector& b,
             std::string solver_type = "lu", std::string pc_type = "ilu");

  /// Compute residual ||Ax - b||
  double residual(const GenericMatrix& A, const GenericVector& x, const GenericVector& b);

  /// Normalize vector according to given normalization type
  double normalize(GenericVector& x, dolfin::NormalizationType normalization_type=normalize_average);

}

#endif
