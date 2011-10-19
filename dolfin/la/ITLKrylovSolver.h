// Copyright (C) 2008-2009 Garth N. Wells
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
// Modified by Dag Lindbo 2008
// Modified by Anders Logg 2008-2011
//
// First added:  2008-07-16
// Last changed: 2011-10-19

#ifdef HAS_MTL4

#ifndef __ITL_KRYLOV_SOLVER_H
#define __ITL_KRYLOV_SOLVER_H

#include <boost/shared_ptr.hpp>
#include <dolfin/common/types.h>
#include "GenericLinearSolver.h"

namespace dolfin
{

  /// Forward declarations
  class GenericMatrix;
  class GenericVector;
  class MTL4Matrix;
  class MTL4Vector;

  /// This class implements Krylov methods for linear systems
  /// of the form Ax = b. It is a wrapper for the Krylov solvers
  /// of ITL.

  class ITLKrylovSolver : public GenericLinearSolver
  {
  public:

    /// Create Krylov solver for a particular method and preconditioner
    ITLKrylovSolver(std::string method = "default",
                    std::string preconditioner = "default");

    /// Destructor
    ~ITLKrylovSolver();

    /// Set operator (matrix)
    void set_operator(const boost::shared_ptr<const GenericMatrix> A);

    /// Set operator (matrix) and preconditioner matrix
    void set_operators(const boost::shared_ptr<const GenericMatrix> A,
                       const boost::shared_ptr<const GenericMatrix> P);

    /// Get operator (matrix)
    const GenericMatrix& get_operator() const;

    /// Solve linear system Ax = b and return number of iterations
    uint solve(GenericVector& x, const GenericVector& b);

    /// Solve linear system Ax = b and return number of iterations
    uint solve(MTL4Vector& x, const MTL4Vector& b);

    /// Solve linear system Ax = b and return number of iterations
    uint solve(const GenericMatrix& A, GenericVector& x, const GenericVector& b);

    /// Return informal string representation (pretty-print)
    std::string str(bool verbose) const;

    /// Return a list of available methods
    static std::vector<std::pair<std::string, std::string> > methods();

    /// Return a list available preconditioners
    static std::vector<std::pair<std::string, std::string> > preconditioners();

    /// Default parameter values
    static Parameters default_parameters();

  private:

    /// Operator (the matrix) as MTL4Matrix
    boost::shared_ptr<const MTL4Matrix> A;

    /// Matrix used to construct the preconditoner
    boost::shared_ptr<const MTL4Matrix> P;

    // Solver type
    std::string method;

    // Preconditioner type
    std::string preconditioner;

  };

}

#endif

#endif
