// Copyright (C) 2007-2008 Anders Logg
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
// Modified by Ola Skavhaug 2008.
//
// First added:  2007-04-30
// Last changed: 2008-08-25

#ifndef __SOLVE_H
#define __SOLVE_H

#include <dolfin/common/types.h>

namespace dolfin
{

  class GenericMatrix;
  class GenericVector;

  /// Solve linear system Ax = b
  void solve(const GenericMatrix& A, GenericVector& x, const GenericVector& b,
             std::string solver_type = "lu", std::string pc_type = "default");

  /// Compute residual ||Ax - b||
  double residual(const GenericMatrix& A, const GenericVector& x, const GenericVector& b);

  /// Normalize vector according to given normalization type
  double normalize(GenericVector& x, std::string normalization_type = "average");

}

#endif
