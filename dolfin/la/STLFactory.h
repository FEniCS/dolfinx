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

#include <boost/shared_ptr.hpp>
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
    boost::shared_ptr<GenericMatrix> create_matrix() const
    {
      boost::shared_ptr<GenericMatrix> A(new STLMatrix);
      return A;
    }

    /// Create empty vector (global)
    boost::shared_ptr<GenericVector> create_vector() const
    {
      boost::shared_ptr<GenericVector> x(new STLVector("global"));
      return x;
    }

    /// Create empty vector (local)
    boost::shared_ptr<GenericVector> create_local_vector() const
    {
      boost::shared_ptr<GenericVector> x(new STLVector("local"));
      return x;
    }

    /// Create empty tensor layout
    boost::shared_ptr<TensorLayout> create_layout(unsigned int rank) const
    {
      boost::shared_ptr<TensorLayout> pattern(new TensorLayout(0, false));
      return pattern;
    }

    /// Create empty linear operator
    boost::shared_ptr<GenericLinearOperator> create_linear_operator() const
    {
      dolfin_error("STLFactory.h",
                   "create linear operator",
                   "Not supported by STL linear algebra backend");
      boost::shared_ptr<GenericLinearOperator> A(new NotImplementedLinearOperator);
      return A;
    }

    /// Create LU solver
    boost::shared_ptr<GenericLUSolver> create_lu_solver(std::string method) const
    {
      dolfin_error("STLFactory",
                   "create LU solver",
                   "LU solver not available for the STL backend");
      boost::shared_ptr<GenericLUSolver> solver;
      return solver;
    }

    /// Create Krylov solver
    boost::shared_ptr<GenericLinearSolver> create_krylov_solver(std::string method,
                                              std::string preconditioner) const
    {
      dolfin_error("STLFactory",
                   "create Krylov solver",
                   "Krylov solver not available for the STL backend");
      boost::shared_ptr<GenericLinearSolver> solver;
      return solver;
    }

    /// Return singleton instance
    static STLFactory& instance()
    { return factory; }

  protected:

    /// Private Constructor
    STLFactory() {}

    // Singleton instance
    static STLFactory factory;

  };
}

#endif
