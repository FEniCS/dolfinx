// Copyright (C) 2006 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg 2006.
//
// First added:  2006-05-31
// Last changed: 2006-08-18

#ifndef __UBLAS_KRYLOV_SOLVER_H
#define __UBLAS_KRYLOV_SOLVER_H

#include <dolfin/common/types.h>
#include "ublas.h"

#include <dolfin/parameter/Parametrized.h>
#include "Preconditioner.h"
#include "KrylovMethod.h"
#include "uBlasKrylovMatrix.h"
#include "uBlasMatrix.h"
#include "uBlasVector.h"
#include "uBlasPreconditioner.h"

namespace dolfin
{

  /// This class implements Krylov methods for linear systems
  /// of the form Ax = b using uBlas data types.

  class uBlasKrylovSolver : public Parametrized
  {
  public:

    /// Create Krylov solver for a particular method and preconditioner
    uBlasKrylovSolver(KrylovMethod method = default_method, Preconditioner pc = default_pc);

    /// Create Krylov solver for a particular preconditioner (set by name)
    uBlasKrylovSolver(Preconditioner pc);

    /// Create Krylov solver for a particular uBlasPreconditioner
    uBlasKrylovSolver(uBlasPreconditioner& pc);

    /// Create Krylov solver for a particular method and uBlasPreconditioner
    uBlasKrylovSolver(KrylovMethod method, uBlasPreconditioner& preconditioner);

    /// Destructor
    ~uBlasKrylovSolver();

    /// Solve linear system Ax = b and return number of iterations (dense matrix)
    uint solve(const uBlasMatrix<ublas_dense_matrix>& A, uBlasVector& x, 
               const uBlasVector& b);

    /// Solve linear system Ax = b and return number of iterations (sparse matrix)
    uint solve(const uBlasMatrix<ublas_sparse_matrix>& A, uBlasVector& x, 
               const uBlasVector& b);
    
    /// Solve linear system Ax = b and return number of iterations (virtual matrix)
    uint solve(const uBlasKrylovMatrix& A, uBlasVector& x, const uBlasVector& b);

  private:

    /// Select solver and solve linear system Ax = b and return number of iterations
    template<class Mat>
    uint solveKrylov(const Mat& A, uBlasVector& x, const uBlasVector& b);

    /// Solve linear system Ax = b using restarted GMRES
    template<class Mat>    
    uint solveGMRES(const Mat& A, uBlasVector& x, const uBlasVector& b, 
                        bool& converged) const;

    /// Solve linear system Ax = b using BiCGStab
    template<class Mat>    
    uint solveBiCGStab(const Mat& A, uBlasVector& x, const uBlasVector& b, 
                        bool& converged) const;
    
    /// Select and create named preconditioner
    void selectPreconditioner(const Preconditioner preconditioner);

    /// Read solver parameters
    void readParameters();

    /// Krylov method
    KrylovMethod method;

    /// Preconditioner
    uBlasPreconditioner* pc;

    /// True if a user has provided a preconditioner
    bool pc_user;

    /// Solver parameters
    real rtol, atol, div_tol;
    uint max_it, restart;
    bool report;

    /// True if we have read parameters
    bool parameters_read;

  };
  //---------------------------------------------------------------------------
  // Implementation of template functions
  //---------------------------------------------------------------------------
  template<class Mat>
  dolfin::uint uBlasKrylovSolver::solveKrylov(const Mat& A, uBlasVector& x, 
        const uBlasVector& b)
  {
    // Check dimensions
    uint M = A.size(0);
    uint N = A.size(1);
    if ( N != b.size() )
      error("Non-matching dimensions for linear system.");

    // Reinitialise x if necessary 
    // FIXME: this erases initial guess
    x.init(b.size());

    // Read parameters if not done
    if ( !parameters_read )
      readParameters();

    // Write a message
    if ( report )
      message("Solving linear system of size %d x %d (uBlas Krylov solver).", M, N);

    // Initialise preconditioner if necessary
    pc->init(A);

    // Choose solver
    bool converged;
    uint iterations;
    switch (method)
    { 
    case gmres:
      iterations = solveGMRES(A, x, b, converged);
      break;
    case bicgstab:
      iterations = solveBiCGStab(A, x, b, converged);
      break;
    case default_method:
      iterations = solveBiCGStab(A, x, b, converged);
      break;
    default:
      warning("Requested Krylov method unknown. Using BiCGStab.");
      iterations = solveBiCGStab(A, x, b, converged);
    }
  
    // Check for convergence
    if( !converged )
      warning("Krylov solver failed to converge.");
    else if ( report )
      message("Krylov solver converged in %d iterations.", iterations);
  
    return iterations; 
  }
  //-----------------------------------------------------------------------------
  template<class Mat> 
  dolfin::uint uBlasKrylovSolver::solveGMRES(const Mat& A,
					   uBlasVector& x, 
					   const uBlasVector& b,
					   bool& converged) const
  {
    // Get underlying uBLAS vectors
    ublas_vector& _x = x.vec(); 
    const ublas_vector& _b = b.vec(); 

    // Get size of system
    const uint size = A.size(0);

    // Create residual vector
    uBlasVector r(size);
    ublas_vector& _r = r.vec(); 

    // Create H matrix and h vector
    ublas_matrix_cmajor_tri H(restart, restart);
    ublas_vector _h(restart+1);

    // Create gamma vector
    ublas_vector _gamma(restart+1);

    // Matrix containing v_k as columns.
    ublas_matrix_cmajor V(size, restart+1);

    // w vector    
    uBlasVector w(size);
    ublas_vector& _w = w.vec(); 

    // Givens vectors
    ublas_vector _c(restart), _s(restart);

    // Miscellaneous storage
    real nu, temp1, temp2, r_norm = 0.0, beta0 = 0;

    converged = false;
    uint iteration = 0;
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
      const real beta = norm_2(_r);
 
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
      uint subiteration = 0;    
      uint j = 0; 
      while (subiteration < restart && iteration < max_it && !converged && r_norm/beta < div_tol) 
      {
        // Compute product w = A*V_j (use r for temporary storage)
        //axpy_prod(A, column(V, j), w, true);
        noalias(_r) = column(V, j);
        A.mult(r, w);

        // Apply preconditioner (use r for temporary storage)
        _r.assign(_w);
        pc->solve(w, r);
  
        for (uint i=0; i <= j; ++i) 
        {
          _h(i)= inner_prod(_w, column(V,i));
          noalias(_w) -= _h(i)*column(V,i);
        }
        _h(j+1) = norm_2(_w);

        // Insert column of V (inserting v_(j+1)
        noalias(column(V,j+1)) = _w/_h(j+1);

        // Apply previous Givens rotations to the "new" column 
        // (this could be improved? - use more uBlas functions. 
        //  The below has been taken from old DOLFIN code.)
        for(uint i=0; i<j; ++i)
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
        // but this gives an error when uBlas debugging is turned onand H 
        // is a triangular matrix
        for(uint i=0; i<j+1; ++i)
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
      ublas::inplace_solve(Htrunc, _g, ublas::upper_tag () );
    
      // x_m = x_0 + V*y
      ublas_matrix_cmajor_range _v( V, ublas::range(0,V.size1()), ublas::range(0,subiteration) );
      axpy_prod(_v, _g, _x, false);
    }
    return iteration;
  }
  //-----------------------------------------------------------------------------
  template<class Mat> 
  dolfin::uint uBlasKrylovSolver::solveBiCGStab(const Mat& A,
					      uBlasVector& x,
					      const uBlasVector& b,
					      bool& converged) const
  {
    // Get uderlying uBLAS vectors
    ublas_vector& _x = x.vec(); 
    const ublas_vector& _b = b.vec(); 

   // Get size of system
    const uint size = A.size(0);

    // Allocate vectors
    uBlasVector r(size), rstar(size), p(size), s(size), v(size), t(size), y(size), z(size);
    ublas_vector& _r = r.vec(); 
    ublas_vector& _rstar = rstar.vec(); 
    ublas_vector& _p = p.vec(); 
    ublas_vector& _s = s.vec(); 
    ublas_vector& _v = v.vec(); 
    ublas_vector& _t = t.vec(); 
    ublas_vector& _y = y.vec(); 
    ublas_vector& _z = z.vec(); 

    real alpha = 1.0, beta = 0.0, omega = 1.0, r_norm = 0.0; 
    real rho_old = 1.0, rho = 1.0;

    // Compute residual r = b -A*x
    //r.assign(b);
    //axpy_prod(A, -x, r, false);
    A.mult(x, r);
    r *= -1.0;
    noalias(_r) += _b;

    const real r0_norm = norm_2(_r);
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
    uint iteration = 0;
    while (iteration < max_it && !converged && r_norm/r0_norm < div_tol) 
    {
      // Set rho_n = rho_n+1
      rho_old = rho; 

      // Compute new rho
      rho = ublas::inner_prod(_r, _rstar); 
      if( fabs(rho) < 1e-25 )
        error("BiCGStab breakdown. rho = %g", rho);

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
