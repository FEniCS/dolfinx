// Copyright (C) 2007-2010 Garth N. Wells
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
// Modified by Dag Lindbo, 2008.
// Modified by Anders Logg, 2008-2009.
// Modified by Kent-Andre Mardal, 2008.
//
// First added:  2007-07-03
// Last changed: 2011-03-24

#ifndef __LU_SOLVER_H
#define __LU_SOLVER_H

#include <string>
#include <boost/scoped_ptr.hpp>
#include "GenericLUSolver.h"

namespace dolfin
{

  // Forward declarations
  class GenericMatrix;
  class GenericVector;

  class LUSolver : public GenericLUSolver
  {

  /// LU solver for the built-in LA backends. The type can be "lu" or
  /// "cholesky". Cholesky is suitable only for symmetric positive-definite
  /// matrices. Cholesky is not yet suppprted for all backends (which will
  /// default to LU.

  public:

    /// Constructor
    LUSolver(std::string type = "lu");

    /// Constructor
    LUSolver(const GenericMatrix& A, std::string type = "lu");

    /// Destructor
    ~LUSolver();

    /// Set operator (matrix)
    void set_operator(const GenericMatrix& A);

    /// Return the operator (matrix)
    const GenericMatrix& get_operator() const;

    /// Solve linear system Ax = b
    uint solve(GenericVector& x, const GenericVector& b);

    /// Solve linear system
    uint solve(const GenericMatrix& A, GenericVector& x, const GenericVector& b);

    /// Default parameter values
    static Parameters default_parameters()
    {
      Parameters p("lu_solver");
      p.add("report", true);
      p.add("same_nonzero_pattern", false);
      p.add("reuse_factorization", false);
      return p;
    }

  private:

    // Solver
    boost::scoped_ptr<GenericLinearSolver> solver;

  };
}

#endif
