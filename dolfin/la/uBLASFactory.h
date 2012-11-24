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
// Modified by Anders Logg 2011-2012
//
// First added:  2007-12-06
// Last changed: 2012-08-20

#ifndef __UBLAS_FACTORY_H
#define __UBLAS_FACTORY_H

#include <string>
#include <boost/shared_ptr.hpp>

#include "uBLASKrylovSolver.h"
#include "uBLASMatrix.h"
#include "uBLASVector.h"
#include "TensorLayout.h"
#include "UmfpackLUSolver.h"
#include "GenericLinearAlgebraFactory.h"

namespace dolfin
{
  // Forward declaration
  class GenericLinearSolver;

  template<typename Mat = ublas_sparse_matrix>
  class uBLASFactory : public GenericLinearAlgebraFactory
  {
  public:

    /// Destructor
    virtual ~uBLASFactory() {}

    /// Create empty matrix
    boost::shared_ptr<GenericMatrix> create_matrix() const
    {
      boost::shared_ptr<GenericMatrix> A(new uBLASMatrix<Mat>);
      return A;
    }

    /// Create empty vector
    boost::shared_ptr<GenericVector> create_vector() const
    {
      boost::shared_ptr<GenericVector> x(new uBLASVector);
      return x;
    }

    /// Create empty vector (local)
    boost::shared_ptr<GenericVector> create_local_vector() const
    {
      boost::shared_ptr<GenericVector> x(new uBLASVector);
      return x;
    }

    /// Create empty tensor layout
    boost::shared_ptr<TensorLayout> create_layout(unsigned int rank) const
    {
      bool sparsity = false;
      if (rank > 1)
        sparsity = true;
      boost::shared_ptr<TensorLayout> pattern(new TensorLayout(0, sparsity));
      return pattern;
    }

    /// Create empty linear operator
    boost::shared_ptr<GenericLinearOperator> create_linear_operator() const
    {
      boost::shared_ptr<GenericLinearOperator> A(new uBLASLinearOperator);
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
        solver(new uBLASKrylovSolver(method, preconditioner));
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
