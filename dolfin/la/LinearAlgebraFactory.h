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
// First added:  2007-11-30
// Last changed: 2011-01-14

#ifndef __LINEAR_ALGEBRA_FACTORY_H
#define __LINEAR_ALGEBRA_FACTORY_H

#include <string>

namespace dolfin
{

  class GenericLinearSolver;
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
    virtual GenericMatrix* create_matrix() const = 0;

    /// Create empty vector (global)
    virtual GenericVector* create_vector() const = 0;

    /// Create empty vector (local)
    virtual GenericVector* create_local_vector() const = 0;

    /// Create empty sparsity pattern (returning zero if not used/needed)
    virtual GenericSparsityPattern* create_pattern() const = 0;

    /// Create LU solver
    virtual GenericLinearSolver* create_lu_solver() const = 0;

    /// Create Krylov solver
    virtual GenericLinearSolver* create_krylov_solver(std::string method,
                                                      std::string pc) const = 0;

  };

}

#endif
