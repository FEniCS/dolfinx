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
// Modified by Anders Logg 2011
//
// First added:  2007-11-30
// Last changed: 2011-10-19

#ifndef __LINEAR_ALGEBRA_FACTORY_H
#define __LINEAR_ALGEBRA_FACTORY_H

#include <string>
#include <vector>
#include <boost/shared_ptr.hpp>

namespace dolfin
{

  class GenericLinearSolver;
  class GenericLUSolver;
  class GenericMatrix;
  class GenericSparsityPattern;
  class GenericVector;

  class LinearAlgebraFactory
  {
    public:

    /// Constructor
    LinearAlgebraFactory() {}

    /// Destructor
    virtual ~LinearAlgebraFactory() {}

    /// Create empty matrix
    virtual boost::shared_ptr<GenericMatrix> create_matrix() const = 0;

    /// Create empty vector (global)
    virtual boost::shared_ptr<GenericVector> create_vector() const = 0;

    /// Create empty vector (local)
    virtual boost::shared_ptr<GenericVector> create_local_vector() const = 0;

    /// Create empty sparsity pattern (returning zero if not used/needed)
    virtual boost::shared_ptr<GenericSparsityPattern> create_pattern(uint primary_dim) const = 0;

    /// Create LU solver
    virtual boost::shared_ptr<GenericLUSolver>
      create_lu_solver(std::string method) const = 0;

    /// Create Krylov solver
    virtual boost::shared_ptr<GenericLinearSolver>
      create_krylov_solver(std::string method,
      std::string preconditioner) const = 0;

    /// Return a list of available LU solver methods.
    /// This function should be overloaded by subclass if non-empty.
    virtual std::vector<std::pair<std::string, std::string> >
      lu_solver_methods() const
    {
      std::vector<std::pair<std::string, std::string> > methods;
      return methods;
    }

    /// Return a list of available Krylov solver methods.
    /// This function should be overloaded by subclass if non-empty.
    virtual std::vector<std::pair<std::string, std::string> >
      krylov_solver_methods() const
    {
      std::vector<std::pair<std::string, std::string> > methods;
      return methods;
    }

    /// Return a list of available preconditioners.
    /// This function should be overloaded by subclass if non-empty.
    virtual std::vector<std::pair<std::string, std::string> >
      krylov_solver_preconditioners() const
    {
      std::vector<std::pair<std::string, std::string> > preconditioners;
      return preconditioners;
    }

  };

}

#endif
