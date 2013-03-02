// Copyright (C) 2008 Dag Lindbo and Garth N. Wells
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
// First added:  2008-08-15
// Last changed: 2011-11-11

#ifndef __CHOLMOD_CHOLESKY_SOLVER_H
#define __CHOLMOD_CHOLESKY_SOLVER_H

#include <boost/shared_ptr.hpp>
#include "GenericLinearSolver.h"

#ifdef HAS_CHOLMOD
extern "C"
{
  #include <cholmod.h>
}
#endif

namespace dolfin
{

  /// Forward declarations
  class GenericVector;
  class GenericLinearOperator;

  /// This class implements the direct solution (Cholesky
  /// factorization) of linear systems of the form Ax = b. Sparse
  /// matrices are solved using CHOLMOD
  /// http://www.cise.ufl.edu/research/sparse/cholmod/ if installed.

  class CholmodCholeskySolver : public GenericLinearSolver
  {

  public:

    /// Constructor
    CholmodCholeskySolver();

    /// Constructor
    CholmodCholeskySolver(boost::shared_ptr<const GenericLinearOperator> A);

    /// Destructor
    ~CholmodCholeskySolver();

    /// Solve the operator (matrix)
    void set_operator(const boost::shared_ptr<const GenericLinearOperator> A)
    {
      dolfin_error("CholmodCholeskySolver.h",
                   "set operator for CHOLMOD Cholesky solver",
                   "Not implemented");
    }

    /// Solve linear system Ax = b for a sparse matrix using CHOLMOD
    virtual std::size_t solve(const GenericLinearOperator& A,
                              GenericVector& x,
                              const GenericVector& b);

    /// Cholesky-factor sparse matrix A if CHOLMOD is installed
    virtual std::size_t factorize(const GenericLinearOperator& A);

    /// Solve factorized system (CHOLMOD).
    virtual std::size_t factorized_solve(GenericVector& x,
                                         const GenericVector& b);

    /// Default parameter values
    static Parameters default_parameters();

  private:

    // Operator (the matrix)
    boost::shared_ptr<const GenericLinearOperator> _A;

#ifdef HAS_CHOLMOD
    // Data for Cholesky factorization of sparse ublas matrix (cholmod only)
    class Cholmod
    {
    public:

      Cholmod();
      ~Cholmod();

      /// Clear data
      void clear();

      /// Initialise with matrix
      void init(long int* Ap, long int* Ai, double* Ax, std::size_t M,
                std::size_t nz);

      /// Factorize
      void factorize();

      /// Factorized solve
      void factorized_solve(double*x, const double* b);

      std::size_t N;
      bool factorized;

    private:

      /// Compute residual: b-Ax
      cholmod_dense* residual(cholmod_dense* x, cholmod_dense* b);

      /// Compute residual norm
      double residual_norm(cholmod_dense* r, cholmod_dense* x,
			                     cholmod_dense* b);

      /// Perform one refinement
      void refine_once(cholmod_dense* x, cholmod_dense* r);

      /// Check status flag returned by an CHOLMOD function
      void check_status(std::string function);

      // CHOLMOD data
      cholmod_sparse* A_chol;
      cholmod_factor* L_chol;
      cholmod_common c;
    };

    Cholmod cholmod;
#endif
  };

}

#endif
