// Copyright (C) 2007 Ola Skavhaug
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
// Modified by Garth N. Wells 2008
// Modified by Anders Logg 2011
//
// First added:  2007-12-06
// Last changed: 2011-10-19

#ifndef __UBLAS_FACTORY_H
#define __UBLAS_FACTORY_H

#include <string>
#include <boost/assign/list_of.hpp>

#include "uBLASKrylovSolver.h"
#include "uBLASMatrix.h"
#include "uBLASVector.h"
#include "SparsityPattern.h"
#include "UmfpackLUSolver.h"
#include "LinearAlgebraFactory.h"

namespace dolfin
{
  // Forward declaration
  class GenericLinearSolver;

  template<typename Mat = ublas_sparse_matrix>
  class uBLASFactory : public LinearAlgebraFactory
  {
  public:

    /// Destructor
    virtual ~uBLASFactory() {}

    /// Create empty matrix
    uBLASMatrix<Mat>* create_matrix() const
    { return new uBLASMatrix<Mat>(); }

    /// Create empty vector
    uBLASVector* create_vector() const
    { return new uBLASVector(); }

    /// Create empty vector (local)
    uBLASVector* create_local_vector() const
    { return new uBLASVector(); }

    /// Create empty sparsity pattern
    SparsityPattern* create_pattern() const
    { return new SparsityPattern(); }

    /// Create LU solver
    UmfpackLUSolver* create_lu_solver(std::string method) const
    { return new UmfpackLUSolver(); }

    /// Create Krylov solver
    GenericLinearSolver* create_krylov_solver(std::string method,
                                              std::string preconditioner) const
    { return new uBLASKrylovSolver(method, preconditioner); }

    /// Return a list of available LU solver methods
    std::vector<std::pair<std::string, std::string> >
    lu_solver_methods() const
    {
      return boost::assign::pair_list_of
        ("default", "default LU solver")
        ("umfpack", "UMFPACK (Unsymmetric MultiFrontal sparse LU factorization)");
    }

    /// Return a list of available Krylov solver methods
    std::vector<std::pair<std::string, std::string> >
    krylov_solver_methods() const
    {
      return uBLASKrylovSolver::methods();
    }

    /// Return a list of available preconditioners
    std::vector<std::pair<std::string, std::string> >
    krylov_solver_preconditioners() const
    {
      return uBLASKrylovSolver::preconditioners();
    }

    /// Return singleton instance
    static uBLASFactory<Mat>& instance()
    { return factory; }

  private:

    // Private Constructor
    uBLASFactory() {}

    // Singleton instance
    static uBLASFactory<Mat> factory;

  };
}

// Initialise static data
template<typename Mat> dolfin::uBLASFactory<Mat> dolfin::uBLASFactory<Mat>::factory;

#endif
