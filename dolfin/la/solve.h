// Copyright (C) 2007-2011 Anders Logg
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
// Modified by Garth N. Wells 2011.
//
// First added:  2007-04-30
// Last changed: 2011-10-19

#ifndef __SOLVE_LA_H
#define __SOLVE_LA_H

#include <map>
#include <string>
#include <vector>
#include <dolfin/common/types.h>

namespace dolfin
{

  // Forward declarations
  class GenericLinearOperator;
  class GenericVector;

  /// Solve linear system Ax = b
  std::size_t solve(const GenericLinearOperator& A, GenericVector& x,
                    const GenericVector& b,
                    std::string method = "lu",
                    std::string preconditioner = "none");

  /// List available linear algebra backends
  void list_linear_algebra_backends();

  /// List available solver methods for current linear algebra backend
  void list_linear_solver_methods();

  /// List available LU methods for current linear algebra backend
  void list_lu_solver_methods();

  /// List available Krylov methods for current linear algebra backend
  void list_krylov_solver_methods();

  /// List available preconditioners for current linear algebra
  /// backend
  void list_krylov_solver_preconditioners();

  /// Return true if a specific linear algebra backend is supported
  bool has_linear_algebra_backend(std::string backend);

  /// Return true if LU method for the current linear algebra backend is
  /// available
  bool has_lu_solver_method(std::string method);

  /// Return true if Krylov method for the current linear algebra
  /// backend is available
  bool has_krylov_solver_method(std::string method);

  /// Return true if Preconditioner for the current linear algebra
  /// backend is available
  bool has_krylov_solver_preconditioner(std::string preconditioner);

  /// Return available linear algebra backends
  std::map<std::string, std::string> linear_algebra_backends();

  /// Return a list of available solver methods for current linear
  /// algebra backend
  std::map<std::string, std::string> linear_solver_methods();

  /// Return a list of available LU methods for current linear algebra
  /// backend
  std::map<std::string, std::string> lu_solver_methods();

  /// Return a list of available Krylov methods for current linear
  /// algebra backend
  std::map<std::string, std::string> krylov_solver_methods();

  /// Return a list of available preconditioners for current linear
  /// algebra backend
  std::map<std::string, std::string> krylov_solver_preconditioners();

  /// Compute residual ||Ax - b||
  double residual(const GenericLinearOperator& A, const GenericVector& x,
                  const GenericVector& b);

  /// Compute norm of vector. Valid norm types are "l2", "l1" and
  /// "linf".
  double norm(const GenericVector& x, std::string norm_type="l2");

  /// Normalize vector according to given normalization type
  double normalize(GenericVector& x,
                   std::string normalization_type = "average");

}

#endif
