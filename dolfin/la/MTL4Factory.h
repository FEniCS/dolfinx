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
// Modified by Anders Logg 2008-2011
//
// First added:  2008-07-06
// Last changed: 2011-10-19

#ifdef HAS_MTL4

#ifndef __MTL4_FACTORY_H
#define __MTL4_FACTORY_H

#include <string>

#include "ITLKrylovSolver.h"
#include "MTL4Matrix.h"
#include "MTL4Vector.h"
#include "GenericSparsityPattern.h"
#include "UmfpackLUSolver.h"
#include "LinearAlgebraFactory.h"

namespace dolfin
{

  class MTL4Factory : public LinearAlgebraFactory
  {
  public:

    /// Destructor
    virtual ~MTL4Factory() {}

    /// Create empty matrix
    MTL4Matrix* create_matrix() const
    { return new MTL4Matrix(); }

    /// Create empty vector (global)
    MTL4Vector* create_vector() const
    { return new MTL4Vector(); }

    /// Create empty vector (local)
    MTL4Vector* create_local_vector() const
    { return new MTL4Vector(); }

    /// Dummy sparsity pattern
    GenericSparsityPattern* create_pattern() const
    { return 0; }

    /// Create LU solver
    UmfpackLUSolver* create_lu_solver(std::string method) const
    { return new UmfpackLUSolver(); }

    /// Create Krylov solver
    ITLKrylovSolver* create_krylov_solver(std::string method,
                                          std::string preconditioner) const
    { return new ITLKrylovSolver(method, preconditioner); }

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
