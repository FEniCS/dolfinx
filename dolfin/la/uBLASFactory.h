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
// Modified by Garth N. Wells, 2008.
//
// First added:  2007-12-06
// Last changed: 2008-05-18

#ifndef __UBLAS_FACTORY_H
#define __UBLAS_FACTORY_H

#include <string>
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
    UmfpackLUSolver* create_lu_solver() const
    { return new UmfpackLUSolver(); }

    /// Create Krylov solver
    GenericLinearSolver* create_krylov_solver(std::string method, std::string pc) const
    //{ return 0; }
    { return new uBLASKrylovSolver(method, pc); }

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
