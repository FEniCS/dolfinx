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
// Last changed: 2011-10-06

#ifdef HAS_PETSC

#ifndef __PETSC_FACTORY_H
#define __PETSC_FACTORY_H

#include "PETScKrylovSolver.h"
#include "PETScLUSolver.h"
#include "PETScMatrix.h"
#include "PETScVector.h"
#include "SparsityPattern.h"
#include "LinearAlgebraFactory.h"

namespace dolfin
{

  class PETScFactory : public LinearAlgebraFactory
  {
  public:

    /// Destructor
    virtual ~PETScFactory() {}

    /// Create empty matrix
    PETScMatrix* create_matrix() const;

    /// Create empty vector (global)
    PETScVector* create_vector() const;

    /// Create empty vector (local)
    PETScVector* create_local_vector() const;

    /// Create empty sparsity pattern
    SparsityPattern* create_pattern() const;

    /// Create LU solver
    PETScLUSolver* create_lu_solver(std::string method) const;

    /// Create Krylov solver
    PETScKrylovSolver* create_krylov_solver(std::string method,
                                            std::string preconditioner) const;

    /// List available LU methods
    std::vector<std::pair<std::string, std::string> > list_lu_methods() const;

    /// List available Krylov methods
    std::vector<std::pair<std::string, std::string> > list_krylov_methods() const;

    /// List available preconditioners
    std::vector<std::pair<std::string, std::string> > list_preconditioners() const;

    /// Return singleton instance
    static PETScFactory& instance()
    { return factory; }

  private:

    /// Private constructor
    PETScFactory() {}
    static PETScFactory factory;

  };

}

#endif

#endif
