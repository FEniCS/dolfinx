// Copyright (C) 2006-2009 Garth N. Wells
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
// Modified by Anders Logg 2006-2012
//
// First added:  2006-05-31
// Last changed: 2012-08-20

#ifndef __UBLAS_KRYLOV_SOLVER_H
#define __UBLAS_KRYLOV_SOLVER_H

#include <set>
#include <string>
#include <boost/shared_ptr.hpp>
#include <dolfin/common/types.h>
#include "ublas.h"
#include "GenericLinearSolver.h"
#include "uBLASLinearOperator.h"
#include "uBLASMatrix.h"
#include "uBLASVector.h"
#include "uBLASPreconditioner.h"

namespace dolfin
{

  class GenericLinearOperator;
  class GenericVector;

  /// This class implements Krylov methods for linear systems
  /// of the form Ax = b using uBLAS data types.

  class uBLASKrylovSolver : public GenericLinearSolver
  {
  public:

    /// Create Krylov solver for a particular method and preconditioner
    uBLASKrylovSolver(std::string method="default",
                      std::string preconditioner="default");

    /// Create Krylov solver for a particular uBLASPreconditioner
    uBLASKrylovSolver(uBLASPreconditioner& pc);

    /// Create Krylov solver for a particular method and uBLASPreconditioner
    uBLASKrylovSolver(std::string method,
                      uBLASPreconditioner& pc);

    /// Destructor
    ~uBLASKrylovSolver();

    /// Solve the operator (matrix)
    void set_operator(const boost::shared_ptr<const GenericLinearOperator> A)
    { set_operators(A, A); }

    /// Set operator (matrix) and preconditioner matrix
    void set_operators(const boost::shared_ptr<const GenericLinearOperator> A,
                       const boost::shared_ptr<const GenericLinearOperator> P)
    { this->A = A; this->P = P; }


    /// Return the operator (matrix)
    const GenericLinearOperator& get_operator() const
    {
      if (!A)
      {
        dolfin_error("uBLASKrylovSolver.cpp",
                     "access operator for uBLAS Krylov solver",
                     "Operator has not been set");
      }
      return *A;
    }

    /// Solve linear system Ax = b and return number of iterations
    unsigned int solve(GenericVector& x, const GenericVector& b);

    /// Solve linear system Ax = b and return number of iterations
    unsigned int solve(const GenericLinearOperator& A, GenericVector& x, const GenericVector& b);

    /// Return a list of available solver methods
    static std::vector<std::pair<std::string, std::string> > methods();

    /// Return a list of available preconditioners
    static std::vector<std::pair<std::string, std::string> > preconditioners();

    /// Default parameter values
    static Parameters default_parameters();

  private:

    /// Select solver and solve linear system Ax = b and return number of iterations
    template<typename MatA, typename MatP>
    unsigned int solve_krylov(const MatA& A,
                      uBLASVector& x,
                      const uBLASVector& b,
                      const MatP& P);

    /// Solve linear system Ax = b using CG
    template<typename Mat>
    unsigned int solveCG(const Mat& A, uBLASVector& x, const uBLASVector& b,
                 bool& converged) const;

    /// Solve linear system Ax = b using restarted GMRES
    template<typename Mat>
    unsigned int solveGMRES(const Mat& A, uBLASVector& x, const uBLASVector& b,
                        bool& converged) const;

    /// Solve linear system Ax = b using BiCGStab
    template<typename Mat>
    unsigned int solveBiCGStab(const Mat& A, uBLASVector& x, const uBLASVector& b,
                        bool& converged) const;

    /// Select and create named preconditioner
    void select_preconditioner(std::string preconditioner);

    /// Read solver parameters
    void read_parameters();

    /// Krylov method
    std::string method;

    /// Preconditioner
    boost::shared_ptr<uBLASPreconditioner> pc;

    /// Solver parameters
    double rtol, atol, div_tol;
    unsigned int max_it, restart;
    bool report;

    /// Operator (the matrix)
    boost::shared_ptr<const GenericLinearOperator> A;

    /// Matrix used to construct the preconditoner
    boost::shared_ptr<const GenericLinearOperator> P;

  };
  //---------------------------------------------------------------------------
  // Implementation of template functions
  //---------------------------------------------------------------------------
  template<typename MatA, typename MatP>
  unsigned int uBLASKrylovSolver::solve_krylov(const MatA& A,
                                               uBLASVector& x,
                                               const uBLASVector& b,
                                               const MatP& P)
  {
    // Check dimensions
    unsigned int M = A.size(0);
    unsigned int N = A.size(1);
    if ( N != b.size() )
    {
      dolfin_error("uBLASKrylovSolver.h",
                   "solve linear system using uBLAS Krylov solver",
                   "Non-matching dimensions for linear system");
    }

    // Reinitialise x if necessary
    if (x.size() != b.size())
    {
      x.resize(b.size());
      x.zero();
    }

    // Read parameters if not done
    read_parameters();

    // Write a message
    if (report)
      info("Solving linear system of size %d x %d (uBLAS Krylov solver).", M, N);

    // Initialise preconditioner if necessary
    pc->init(P);

    // Choose solver and solve
    bool converged = false;
    unsigned int iterations = 0;
    if (method == "cg")
      iterations = solveCG(A, x, b, converged);
    else if (method == "gmres")
      iterations = solveGMRES(A, x, b, converged);
    else if (method == "bicgstab")
      iterations = solveBiCGStab(A, x, b, converged);
    else if (method == "default")
      iterations = solveBiCGStab(A, x, b, converged);
    else
    {
      dolfin_error("uBLASKrylovSolver.h",
                   "solve linear system using uBLAS Krylov solver",
                   "Requested Krylov method (\"%s\") is unknown", method.c_str());
    }

    // Check for convergence
    if (!converged)
    {
      bool error_on_nonconvergence = parameters["error_on_nonconvergence"];
      if (error_on_nonconvergence)
      {
        dolfin_error("uBLASKrylovSolver.h",
                     "solve linear system using uBLAS Krylov solver",
                     "Solution failed to converge");
      }
      else
        warning("uBLAS Krylov solver failed to converge.");
    }
    else if (report)
      info("Krylov solver converged in %d iterations.", iterations);

    return iterations;
  }
  //-----------------------------------------------------------------------------
  template<typename Mat>
  unsigned int uBLASKrylovSolver::solveCG(const Mat& A,
                                          uBLASVector& x,
                                          const uBLASVector& b,
                                          bool& converged) const
  {
    warning("Conjugate-gradient method not yet programmed for uBLASKrylovSolver. Using GMRES.");
    return solveGMRES(A, x, b, converged);
  }
  //-----------------------------------------------------------------------------
  template<typename Mat>
  unsigned int uBLASKrylovSolver::solveGMRES(const Mat& A, uBLASVector& x,
                                             const uBLASVector& b,
                                             bool& converged) const
  {
    // Get underlying uBLAS vectors
    ublas_vector& _x = x.vec();
    const ublas_vector& _b = b.vec();

    // Get size of system
    const unsigned int size = A.size(0);

    // Create residual vector
    uBLASVector r(size);
    ublas_vector& _r = r.vec();

    // Create H matrix and h vector
    ublas_matrix_cmajor_tri H(restart, restart);
    ublas_vector _h(restart+1);

    // Create gamma vector
    ublas_vector _gamma(restart+1);

    // Matrix containing v_k as columns.
    ublas_matrix_cmajor V(size, restart+1);

    // w vector
    uBLASVector w(size);
    ublas_vector& _w = w.vec();

    // Givens vectors
    ublas_vector _c(restart), _s(restart);

    // Miscellaneous storage
    double nu, temp1, temp2, r_norm = 0.0, beta0 = 0;

    converged = false;
    unsigned int iteration = 0;
    while (iteration < max_it && !converged)
    {
      // Compute residual r = b -A*x
      //noalias(r) = b;
      //axpy_prod(A, -x, r, false);
      A.mult(x, r);
      _r *= -1.0;
      noalias(_r) += _b;

      // Apply preconditioner (use w for temporary storage)
      _w.assign(_r);
      pc->solve(r, w);

      // L2 norm of residual (for most recent restart)
      const double beta = norm_2(_r);

     // Save intial residual (from restart 0)
     if(iteration == 0)
       beta0 = beta;

     if(beta < atol)
     {
       converged = true;
       return iteration;
     }

      // Intialise gamma
      _gamma.clear();
      _gamma(0) = beta;

      // Create first column of V
      noalias(column(V, 0)) = _r/beta;

      // Modified Gram-Schmidt procedure
      unsigned int subiteration = 0;
      unsigned int j = 0;
      while (subiteration < restart && iteration < max_it && !converged && r_norm/beta < div_tol)
      {
        // Compute product w = A*V_j (use r for temporary storage)
        //axpy_prod(A, column(V, j), w, true);
        noalias(_r) = column(V, j);
        A.mult(r, w);

        // Apply preconditioner (use r for temporary storage)
        _r.assign(_w);
        pc->solve(w, r);

        for (unsigned int i=0; i <= j; ++i)
        {
          _h(i)= inner_prod(_w, column(V,i));
          noalias(_w) -= _h(i)*column(V,i);
        }
        _h(j+1) = norm_2(_w);

        // Insert column of V (inserting v_(j+1)
        noalias(column(V,j+1)) = _w/_h(j+1);

        // Apply previous Givens rotations to the "new" column
        // (this could be improved? - use more uBLAS functions.
        //  The below has been taken from old DOLFIN code.)
        for(unsigned int i=0; i<j; ++i)
        {
          temp1 = _h(i);
          temp2 = _h(i+1);
          _h(i)   = _c(i)*temp1 - _s(i)*temp2;
          _h(i+1) = _s(i)*temp1 + _c(i)*temp2 ;
        }

        // Compute new c_i and s_i
        nu = sqrt( _h(j)*_h(j) + _h(j+1)*_h(j+1) );

        // Direct access to c & s below leads to some strange compiler errors
        // when using vector expressions and noalias(). By using "subrange",
        // we are working with vector expressions rather than reals
        //c(j) =  h(j)/nu;
        //s(j) = -h(j+1)/nu;
        subrange(_c, j,j+1) =  subrange(_h, j,j+1)/nu;
        subrange(_s, j,j+1) = -subrange(_h, j+1,j+2)/nu;

        // Apply new rotation to last column
        _h(j)   = _c(j)*_h(j) - _s(j)*_h(j+1);
        _h(j+1) = 0.0;

        // Apply rotations to gamma
        temp1 = _c(j)*_gamma(j) - _s(j)*_gamma(j+1);
        _gamma(j+1) = _s(j)*_gamma(j) + _c(j)*_gamma(j+1);
        _gamma(j) = temp1;
        r_norm = fabs(_gamma(j+1));

        // Add h to H matrix. Would ne nice to use
        //   noalias(column(H, j)) = subrange(h, 0, restart);
        // but this gives an error when uBLAS debugging is turned onand H
        // is a triangular matrix
        for(unsigned int i=0; i<j+1; ++i)
          H(i,j) = _h(i);

        // Check for convergence
        if( r_norm/beta0 < rtol || r_norm < atol )
          converged = true;

        ++iteration;
        ++subiteration;
        ++j;
      }

      // Eliminate extra rows and columns (this does not resize or copy, just addresses a range)
      ublas_matrix_cmajor_tri_range Htrunc(H, ublas::range(0,subiteration), ublas::range(0,subiteration) );
      ublas_vector_range _g(_gamma, ublas::range(0,subiteration));

      // Solve triangular system H*g and return result in g
      ublas::inplace_solve(Htrunc, _g, ublas::upper_tag ());

      // x_m = x_0 + V*y
      ublas_matrix_cmajor_range _v( V, ublas::range(0,V.size1()), ublas::range(0,subiteration) );
      axpy_prod(_v, _g, _x, false);
    }
    return iteration;
  }
  //-----------------------------------------------------------------------------
  template<typename Mat>
  unsigned int uBLASKrylovSolver::solveBiCGStab(const Mat& A,
                                                uBLASVector& x,
                                                const uBLASVector& b,
                                                bool& converged) const
  {
    // Get uderlying uBLAS vectors
    ublas_vector& _x = x.vec();
    const ublas_vector& _b = b.vec();

   // Get size of system
    const unsigned int size = A.size(0);

    // Allocate vectors
    uBLASVector r(size), rstar(size), p(size), s(size), v(size), t(size), y(size), z(size);
    ublas_vector& _r = r.vec();
    ublas_vector& _rstar = rstar.vec();
    ublas_vector& _p = p.vec();
    ublas_vector& _s = s.vec();
    ublas_vector& _v = v.vec();
    ublas_vector& _t = t.vec();
    ublas_vector& _y = y.vec();
    ublas_vector& _z = z.vec();

    double alpha = 1.0, beta = 0.0, omega = 1.0, r_norm = 0.0;
    double rho_old = 1.0, rho = 1.0;

    // Compute residual r = b -A*x
    //r.assign(b);
    //axpy_prod(A, -x, r, false);
    A.mult(x, r);
    r *= -1.0;
    noalias(_r) += _b;

    const double r0_norm = norm_2(_r);
    if( r0_norm < atol )
    {
      converged = true;
      return 0;
    }

    // Initialise r^star, v and p
    _rstar.assign(_r);
    _v.clear();
    _p.clear();

    // Apply preconditioner to r^start. This is a trick to avoid problems in which
    // (r^start, r) = 0  after the first iteration (such as PDE's with homogeneous
    // Neumann bc's and no forcing/source term.
    pc->solve(rstar, r);

    // Right-preconditioned Bi-CGSTAB

    // Start iterations
    converged = false;
    unsigned int iteration = 0;
    while (iteration < max_it && !converged && r_norm/r0_norm < div_tol)
    {
      // Set rho_n = rho_n+1
      rho_old = rho;

      // Compute new rho
      rho = ublas::inner_prod(_r, _rstar);
      if( fabs(rho) < 1e-25 )
      {
        dolfin_error("uBLASKrylovSolver.h",
                     "solve linear system using uBLAS BiCGStab solver",
                     "Solution failed to converge, rho = %g", rho);
      }

      beta = (rho/rho_old)*(alpha/omega);

      // p = r1 + beta*p - beta*omega*A*p
      p *= beta;
      noalias(_p) += _r - beta*omega*_v;

      // My = p
      pc->solve(y, p);

      // v = A*y
      //axpy_prod(A, y, v, true);
      A.mult(y, v);

      // alpha = (r, rstart) / (v, rstar)
      alpha = rho/ublas::inner_prod(_v, _rstar);

      // s = r - alpha*v
      noalias(_s) = _r - alpha*_v;

      // Mz = s
      pc->solve(z, s);

      // t = A*z
      //axpy_prod(A, z, t, true);
      A.mult(z, t);

      // omega = (t, s) / (t,t)
      omega = ublas::inner_prod(_t, _s)/ublas::inner_prod(_t, _t);

      // x = x + alpha*p + omega*s
      noalias(_x) += alpha*_y + omega*_z;

      // r = s - omega*t
      noalias(_r) = _s - omega*_t;

      // Compute norm of the residual and check for convergence
      r_norm = norm_2(_r);
      if( r_norm/r0_norm < rtol || r_norm < atol)
        converged = true;

      ++iteration;
    }

    return iteration;
  }
  //-----------------------------------------------------------------------------
}

#endif
