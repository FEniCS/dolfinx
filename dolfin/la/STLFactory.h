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
// Modified by Anders Logg 2008-2012
// Modified by Garth N. Wells 2012
//
// First added:  2008-05-21
// Last changed: 2012-08-20

#ifndef __DOLFIN_STL_FACTORY_H
#define __DOLFIN_STL_FACTORY_H

#include <memory>
#include <dolfin/log/log.h>
#include "GenericLinearAlgebraFactory.h"
#include "STLMatrix.h"
#include "STLVector.h"
#include "TensorLayout.h"
#include "Vector.h"

namespace dolfin
{

  class STLFactory: public GenericLinearAlgebraFactory
  {
  public:

    /// Destructor
    virtual ~STLFactory() {}

    /// Create empty matrix
    std::shared_ptr<GenericMatrix> create_matrix(MPI_Comm comm) const
    { return  std::make_shared<STLMatrix>(); }

    /// Create empty vector
    std::shared_ptr<GenericVector> create_vector(MPI_Comm comm) const
    { return std::make_shared<STLVector>(); }

    /// Create empty tensor layout
    std::shared_ptr<TensorLayout> create_layout(std::size_t rank) const
    {
      return std::make_shared<TensorLayout>(0, TensorLayout::Sparsity::DENSE);
    }

    /// Create empty linear operator
    std::shared_ptr<GenericLinearOperator> create_linear_operator() const
    {
      dolfin_error("STLFactory.h",
                   "create linear operator",
                   "Not supported by STL linear algebra backend");
      std::shared_ptr<GenericLinearOperator>
        A(new NotImplementedLinearOperator);
      return A;
    }

    /// Create LU solver
    std::shared_ptr<GenericLUSolver>
      create_lu_solver(MPI_Comm comm, std::string method) const
    {
      dolfin_error("STLFactory",
                   "create LU solver",
                   "LU solver not available for the STL backend");
      std::shared_ptr<GenericLUSolver> solver;
      return solver;
    }

    /// Create Krylov solver
    std::shared_ptr<GenericLinearSolver>
      create_krylov_solver(MPI_Comm comm,
                           std::string method,
                           std::string preconditioner) const
    {
      dolfin_error("STLFactory",
                   "create Krylov solver",
                   "Krylov solver not available for the STL backend");
      std::shared_ptr<GenericLinearSolver> solver;
      return solver;
    }

    /// Return singleton instance
    static STLFactory& instance()
    { return factory; }

  protected:

    // Private Constructor
    STLFactory() {}

    // Singleton instance
    static STLFactory factory;

  };
}

#endif
