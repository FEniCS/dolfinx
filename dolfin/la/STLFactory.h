// Copyright (C) 2007 Ilmar Wilbers
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
// First added:  2008-05-21
// Last changed: 2011-11-11

#ifndef __STL_FACTORY_H
#define __STL_FACTORY_H

#include "STLMatrix.h"
#include "uBLASVector.h"
#include "GenericSparsityPattern.h"
#include "LinearAlgebraFactory.h"

namespace dolfin
{

  class STLFactory: public LinearAlgebraFactory
  {
  public:

    /// Destructor
    virtual ~STLFactory() {}

    /// Create empty matrix
    STLMatrix* create_matrix() const
    { return new STLMatrix(); }

    /// Create empty vector (global)
    uBLASVector* create_vector() const
    { return new uBLASVector(); }

    /// Create empty vector (local)
    uBLASVector* create_local_vector() const
    { return new uBLASVector(); }

    /// Create empty sparsity pattern
    GenericSparsityPattern* create_pattern() const
    { return 0; }

    /// Create LU solver
    GenericLinearSolver* create_lu_solver(std::string method) const
    {
      dolfin_error("STLFactory",
                   "create LU solver",
                   "LU solver not available for the STL backend");
      return 0;
    }

    /// Create Krylov solver
    GenericLinearSolver* create_krylov_solver(std::string method,
                                              std::string preconditioner) const
    {
      dolfin_error("STLFactory",
                   "create Krylov solver",
                   "Krylov solver not available for the STL backend");
      return 0;
    }

    /// Return singleton instance
    static STLFactory& instance()
    { return factory; }

  private:

    /// Private Constructor
    STLFactory() {}

    // Singleton instance
    static STLFactory factory;

  };
}

#endif
