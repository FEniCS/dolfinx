// Copyright (C) 2006-2010 Garth N. Wells
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
// Modified by Anders Logg 2006, 2012
// Modified by Dag Lindbo 2008
//
// First added:  2006-05-31
// Last changed: 2012-08-20

#ifndef __UMFPACK_LU_SOLVER_H
#define __UMFPACK_LU_SOLVER_H

#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include <dolfin/common/types.h>
#include "GenericLUSolver.h"

namespace dolfin
{

  /// Forward declarations
  class GenericVector;
  class GenericLinearOperator;
  class uBLASVector;
  class uBLASKrylovMatrix;
  template<typename Mat> class uBLASMatrix;

  /// This class implements the direct solution (LU factorization) of
  /// linear systems of the form Ax = b using UMFPACK
  /// (http://www.cise.ufl.edu/research/sparse/umfpack/) if installed.

  class UmfpackLUSolver : public GenericLUSolver
  {

  public:

    /// Constructor
    UmfpackLUSolver();

    /// Constructor
    UmfpackLUSolver(boost::shared_ptr<const GenericLinearOperator> A);

    /// Destructor
    ~UmfpackLUSolver();

    /// Set operator (matrix)
    void set_operator(const boost::shared_ptr<const GenericLinearOperator> A);

    /// Return the operator (matrix)
    const GenericLinearOperator& get_operator() const;

    /// Solve linear system Ax = b for a sparse matrix using UMFPACK if installed
    uint solve(GenericVector& x, const GenericVector& b);

    /// Solve linear system
    uint solve(const GenericLinearOperator& A, GenericVector& x, const GenericVector& b);

    /// Default parameter values
    static Parameters default_parameters();

  private:

    // Perform symbolic factorisation
    void symbolic_factorize();

    /// LU factorisation
    void numeric_factorize();

    /// Solve factorized system (UMFPACK).
    uint solve_factorized(GenericVector& x, const GenericVector& b) const;

    // Return pointer to symbolic factorisation
    static boost::shared_ptr<void> umfpack_factorize_symbolic(uint M, uint N,
                                                         const std::size_t* Ap,
                                                         const std::size_t* Ai,
                                                         const double* Ax);

    // Return pointer to the numerical factorisation
    static boost::shared_ptr<void> umfpack_factorize_numeric(const std::size_t* Ap,
                                           const std::size_t* Ai,
                                           const double* Ax, void* symbolic);

    static void umfpack_solve(const std::size_t* Ap, const std::size_t* Ai,
                              const double* Ax, double* x, const double* b,
                              void* numeric);

    /// Check status flag returned by an UMFPACK function
    static void umfpack_check_status(long int status, std::string function);

    // UMFPACK data (note that boost::scoped_ptr cannot hold a void pointer)
    boost::shared_ptr<void> symbolic;
    boost::shared_ptr<void> numeric;

    // Operator (the matrix)
    boost::shared_ptr<const GenericLinearOperator> A;

  };

}

#endif
