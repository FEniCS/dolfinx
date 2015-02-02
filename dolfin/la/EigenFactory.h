// Copyright (C) 2015 Chris Richardson and Garth Wells
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
// First added:  2015-02-01

#ifndef __EIGEN_FACTORY_H
#define __EIGEN_FACTORY_H

#include <memory>
#include <string>

#include <dolfin/log/log.h>
// #include "EigenKrylovSolver.h"
#include "EigenMatrix.h"
#include "EigenVector.h"
#include "TensorLayout.h"
#include "GenericLinearAlgebraFactory.h"

namespace dolfin
{
  // Forward declaration
  class GenericLinearSolver;

  class EigenFactory : public GenericLinearAlgebraFactory
  {
  public:

    /// Destructor
    virtual ~EigenFactory() {}

    /// Create empty matrix
    std::shared_ptr<GenericMatrix> create_matrix() const
    {
      std::shared_ptr<GenericMatrix> A(new EigenMatrix);
      return A;
    }

    /// Create empty vector
    std::shared_ptr<GenericVector> create_vector() const
    {
      std::shared_ptr<GenericVector> x(new EigenVector);
      return x;
    }

    /// Create empty tensor layout
    std::shared_ptr<TensorLayout> create_layout(std::size_t rank) const
    {
      bool sparsity = false;
      if (rank > 1)
        sparsity = true;
      std::shared_ptr<TensorLayout> pattern(new TensorLayout(0, sparsity));
      return pattern;
    }

    /// Create empty linear operator
    std::shared_ptr<GenericLinearOperator> create_linear_operator() const
    {
      std::shared_ptr<GenericLinearOperator> A;//(new EigenLinearOperator);
      return A;
    }

    /// Create LU solver
    std::shared_ptr<GenericLUSolver> create_lu_solver(std::string method) const
    {
      return std::shared_ptr<GenericLUSolver>();
    }

    /// Create Krylov solver
    std::shared_ptr<GenericLinearSolver>
    create_krylov_solver(std::string method,
                         std::string preconditioner) const
    {
      std::shared_ptr<GenericLinearSolver>
        solver;//(new EigenKrylovSolver(method, preconditioner));
      return solver;
    }

    /// Return a list of available LU solver methods
    std::vector<std::pair<std::string, std::string> >
      lu_solver_methods() const
    {
      std::vector<std::pair<std::string, std::string> > methods;
      methods.push_back(std::make_pair("default",
                                       "default LU solver"));
      return methods;
    }

    /// Return a list of available Krylov solver methods
    std::vector<std::pair<std::string, std::string> >
    krylov_solver_methods() const
    {
      std::vector<std::pair<std::string, std::string> > methods;
      return methods;
      //return EigenKrylovSolver::methods();
    }

    /// Return a list of available preconditioners
    std::vector<std::pair<std::string, std::string> >
    krylov_solver_preconditioners() const
    {
      std::vector<std::pair<std::string, std::string> > methods;
      return methods;
      //      return EigenKrylovSolver::preconditioners();
    }

    /// Return singleton instance
    static EigenFactory& instance()
    { return factory; }

  private:

    // Private Constructor
    EigenFactory() {}

    // Singleton instance
    static EigenFactory factory;
  };

}
#endif
