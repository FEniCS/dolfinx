// Copyright (C) 2008 Dag Lindbo
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
// Modified by Anders Logg 2008-2012
//
// First added:  2008-07-06
// Last changed: 2012-08-20

#ifdef HAS_MTL4

#ifndef __MTL4_FACTORY_H
#define __MTL4_FACTORY_H

#include <string>
#include <boost/shared_ptr.hpp>

#include "ITLKrylovSolver.h"
#include "MTL4Matrix.h"
#include "MTL4Vector.h"
#include "TensorLayout.h"
#include "UmfpackLUSolver.h"
#include "GenericLinearAlgebraFactory.h"

namespace dolfin
{

  class MTL4Factory : public GenericLinearAlgebraFactory
  {
  public:

    /// Destructor
    virtual ~MTL4Factory() {}

    /// Create empty matrix
    boost::shared_ptr<GenericMatrix> create_matrix() const
    {
      boost::shared_ptr<GenericMatrix> A(new MTL4Matrix);
      return A;
    }

    /// Create empty vector (global)
    boost::shared_ptr<GenericVector> create_vector() const
    {
      boost::shared_ptr<GenericVector> x(new MTL4Vector);
      return x;
    }

    /// Create empty vector (local)
    boost::shared_ptr<GenericVector> create_local_vector() const
    {
      boost::shared_ptr<GenericVector> x(new MTL4Vector);
      return x;
    }

    /// Create empty tensor layout
    boost::shared_ptr<TensorLayout> create_layout(uint rank) const
    {
      boost::shared_ptr<TensorLayout> pattern(new TensorLayout(0, false));
      return pattern;
    }

    /// Create empty Krylov matrix
    boost::shared_ptr<GenericLinearOperator> create_linear_operator() const
    {
      dolfin_error("MTL4Factory.h",
                   "create Krylov matrix",
                   "Not supported by MTL4 linear algebra backend");
      boost::shared_ptr<GenericLinearOperator> A(new NotImplementedLinearOperator);
      return A;
    }

    /// Create LU solver
    boost::shared_ptr<GenericLUSolver> create_lu_solver(std::string method) const
    {
      boost::shared_ptr<GenericLUSolver> solver(new UmfpackLUSolver);
      return solver;
    }

    /// Create Krylov solver
    boost::shared_ptr<GenericLinearSolver> create_krylov_solver(std::string method,
                                              std::string preconditioner) const
    {
      boost::shared_ptr<GenericLinearSolver>
        solver(new ITLKrylovSolver(method, preconditioner));
      return solver;
    }

    /// Return a list of available LU solver methods
    std::vector<std::pair<std::string, std::string> >
      lu_solver_methods() const
    {
      std::vector<std::pair<std::string, std::string> > methods;
      methods.push_back(std::make_pair("default",
                                       "default LU solver"));
      methods.push_back(std::make_pair("umfpack",
                                       "UMFPACK (Unsymmetric MultiFrontal sparse LU factorization)"));
      return methods;
    }

    /// Return a list of available Krylov solver methods
    std::vector<std::pair<std::string, std::string> >
      krylov_solver_methods() const
    { return ITLKrylovSolver::methods(); }

    /// Return a list of available preconditioners
    std::vector<std::pair<std::string, std::string> >
      krylov_solver_preconditioners() const
    { return ITLKrylovSolver::preconditioners(); }

    // Return singleton instance
    static MTL4Factory& instance()
    { return factory; }

  private:

    // Private constructor
    MTL4Factory() {}

    // Singleton instance
    static MTL4Factory factory;

  };
}

#endif

#endif
