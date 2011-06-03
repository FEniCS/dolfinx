// Copyright (C) 2007-2009 Garth N. Wells
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
// Modified by Ola Skavhaug, 2008.
// Modified by Anders Logg, 2008.
//
// First added:  2007-07-03
// Last changed: 2011-03-24

#ifndef __KRYLOV_SOLVER_H
#define __KRYLOV_SOLVER_H

#include <string>
#include <boost/scoped_ptr.hpp>
#include "GenericLinearSolver.h"

namespace dolfin
{

  class GenericMatrix;
  class GenericVector;

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

    /// Set operator (matrix)
    void set_operator(const GenericMatrix& A);

    /// Set operator (matrix) and preconditioner matrix
    void set_operators(const GenericMatrix& A, const GenericMatrix& P);

    /// Get operator (matrix)
    const GenericMatrix& get_operator() const;

    /// Solve linear system Ax = b
    uint solve(GenericVector& x, const GenericVector& b);

    /// Solve linear system Ax = b
    uint solve(const GenericMatrix& A, GenericVector& x, const GenericVector& b);

    /// Default parameter values
    static Parameters default_parameters();

  private:

    // Solver
    boost::scoped_ptr<GenericLinearSolver> solver;

  };
}

#endif
