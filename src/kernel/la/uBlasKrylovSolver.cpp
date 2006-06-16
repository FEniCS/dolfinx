// Copyright (C) 2006 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-05-31
// Last changed: 2006-06-06


#include <dolfin/dolfin_log.h>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>

#include <dolfin/uBlasKrylovSolver.h>


using namespace dolfin;

//-----------------------------------------------------------------------------
uBlasKrylovSolver::uBlasKrylovSolver() : Parametrized(), type(default_solver)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
uBlasKrylovSolver::uBlasKrylovSolver(Type solver) : Parametrized(), type(solver)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
uBlasKrylovSolver::~uBlasKrylovSolver()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
dolfin::uint uBlasKrylovSolver::solve(const uBlasSparseMatrix& A, DenseVector& x, 
    const DenseVector& b)
{
  // Check dimensions
  uint M = A.size(0);
  uint N = A.size(1);
  if ( N != b.size() )
    dolfin_error("Non-matching dimensions for linear system.");

  // Reinitialise x if necessary FIXME: this erases initial guess
  x.init(b.size());

  // Write a message
  if ( get("Krylov report") )
    dolfin_info("Solving linear system of size %d x %d (uBlas Krylov solver).", M, N);

  // Choose solver
  bool converged;
  uint iterations;
  switch (type)
  { 
  case gmres:
    iterations = gmresSolver(A, x, b, converged);
    break;
  case bicgstab:
    iterations = bicgstabSolver(A, x, b, converged);
    break;
  case default_solver:
    iterations = bicgstabSolver(A, x, b, converged);
    break;
  default:
    dolfin_warning("Requested solver type unknown. USing BiCGStab.");
    iterations = bicgstabSolver(A, x, b, converged);
  }

  // Check for convergence
  if( !converged )
    dolfin_warning("Krylov solver failed to converge.");
  else
    if ( get("Krylov report") )
      dolfin_info("Krylov solver converged in %d iterations.", iterations);

  return iterations; 
}
//-----------------------------------------------------------------------------
dolfin::uint uBlasKrylovSolver::gmresSolver(const uBlasSparseMatrix& A, DenseVector& x, 
    const DenseVector& b, bool& converged)
{

  dolfin_warning("GMRES solver for uBlas data types has not been optimised.");
  dolfin_warning("Preconditioning has not yet been implemented, so convergence may be poor.");

  namespace ublas = boost::numeric::ublas;
  typedef ublas::vector<double> ublas_vector;
  typedef ublas::matrix<double,ublas::column_major> ublas_matrix;
  typedef ublas::matrix_range<ublas_matrix> ublas_matrix_range;
  typedef ublas::vector_range<ublas_vector> ublas_vector_range;

  // Get tolerances
  const real rtol    = get("Krylov relative tolerance");
  const real atol    = get("Krylov absolute tolerance");
  const real div_tol = get("Krylov divergence limit");
  const uint max_it  = get("Krylov maximum iterations");
  const uint restart = get("Krylov GMRES restart");

  // Get size of system
  const uint size = A.size(0);

  // Allocate residual vector
  ublas_vector r(size);

  // Create H matrix
  ublas_matrix H(restart+1, restart);

  // Create gamma vector
  ublas_vector gamma(restart+1);

  // Matrix containing v_k as columns    
  ublas_matrix V(size, 1);

  // w vector    
  ublas_vector w(size);

  // y vector    
  ublas_vector y(restart+1);

  // Givens vectors
  ublas_vector c(restart+1);
  ublas_vector s(restart+1);

  // Miscellaneous storage
  real nu, temp1, temp2, r_norm = 0.0, beta0 = 0;

//  bool converged = false;
  converged = false;
  uint iteration = 0;
  while (iteration < max_it && !converged) 
  {
    
    // Clear data for restart
    H.clear();  
    V.clear();  
    gamma.clear();
    c.clear();
    s.clear();

    // Compute residual r = b -A*x
    noalias(r) = b;
    axpy_prod(A, -x, r, false); 

    // L2 norm of residual (for most recent restart)
    const real beta = norm_2(r);

    if(iteration == 0)
      beta0 = beta;

    if(beta < atol)
    { 
      converged = true;
      return iteration;
    }

    // Intialise gamma
    gamma(0) = beta;

    // Create first column of V
    noalias(column(V, 0)) = r/beta;

    // Modified Gram-Schmidt procedure
    uint subiteration = 0;    
    uint j = 0; 
    while (subiteration < restart && iteration < max_it && !converged && r_norm/beta < div_tol) 
    {
      axpy_prod(A, column(V,j), w, true);    
      for (uint i=0; i <= j; ++i) 
      {
        H(i,j)= inner_prod(w, column(V,i));
        w -= H(i,j)*column(V,i);
      }
      H(j+1,j) = norm_2(w);

      // Add column to V and insert v_(j+1)
      V.resize(V.size1(), j+2, true);  
      noalias(column(V,j+1)) = w/H(j+1,j);

      // Perform Givens rotations (this should be improved - use more uBlas functions)
      for(uint i=0; i<j; ++i)
      {
        temp1 = H(i,j);        
        temp2 = H(i+1,j);        
        H(i,j)   = c(i)*temp1 - s(i)*temp2;      
        H(i+1,j) = s(i)*temp1 + c(i)*temp2 ;     
      }
      nu = sqrt( H(j,j)*H(j,j) + H(j+1,j)*H(j+1,j) );
      c(j) =  H(j,j)/nu; 
      s(j) = -H(j+1,j)/nu; 

      H(j,j)   = c(j)*H(j,j) - s(j)*H(j+1,j);
      H(j+1,j) = 0.0;

      temp1 = c(j)*gamma(j) - s(j)*gamma(j+1);
      gamma(j+1) = s(j)*gamma(j) + c(j)*gamma(j+1); 
      gamma(j) = temp1; 
      r_norm = fabs(gamma(j+1));

      // Check for convergence
      if( r_norm/beta0 < rtol || r_norm < atol )
        converged = true;

      ++iteration;
      ++subiteration;
      ++j;
    }

    // Eliminate extra rows and columns (this does not resize or copy, just addresses a range)
    ublas_matrix_range h(H, ublas::range(0,subiteration), ublas::range(0,subiteration) );
    ublas_vector_range g(gamma, ublas::range(0,subiteration));
    ublas_vector_range ys(y, ublas::range(0,subiteration));

    // Solve triangular system y = H*gamma
    noalias(ys) = ublas::solve(h, g, ublas::upper_tag () );
    
    // x_m = x_0 + V*y
    ublas_matrix_range v(V, ublas::range(0,V.size1()), ublas::range(0,subiteration) );
    axpy_prod(v, ys, x, false);

  }  

  return iteration;
}
//-----------------------------------------------------------------------------
dolfin::uint uBlasKrylovSolver::bicgstabSolver(const uBlasSparseMatrix& A, DenseVector& x, 
    const DenseVector& b, bool& converged)
{

  dolfin_warning("Preconditioning has not yet been implemented for the uBlas BiCGStab solver.");

  namespace ublas = boost::numeric::ublas;
  typedef ublas::vector<double> ublas_vector;
  typedef ublas::matrix<double> ublas_matrix;

  // Get tolerances
  const real rtol    = get("Krylov relative tolerance");
  const real atol    = get("Krylov absolute tolerance");
  const real div_tol = get("Krylov divergence limit");
  const uint max_it  = get("Krylov maximum iterations");

  // Get size of system
  const uint size = A.size(0);

  // Allocate vectors
  ublas_vector r0(size);
  ublas_vector r1(size);
  ublas_vector rstar(size);
  ublas_vector p(size);
  ublas_vector s(size);

  ublas_vector Ap(size);
  ublas_vector As(size);
  ublas_vector vtemp(size);

  // Compute residual r = b -A*x
  r1.assign(b);
  axpy_prod(A, -x, r1, false); 

  const real r0_norm = norm_2(r1);
  if( r0_norm < atol )  
  {
    converged = true;    
    return 0;
  }  

  p.assign(r1);

  // What's a good value here for r^star?
  rstar.assign(b);

  real alpha = 0.0, beta = 0.0, omega = 0.0, r_norm = 0.0; 
  
  // Save inner products (r0, r*) and (r1, r*) 
  real r_rstar0 = 0.0;
  real r_rstar1;
  
  r_rstar1 = ublas::inner_prod(r1,rstar);

  // Start iterations
  converged = false;
  uint iteration = 0;
  while (iteration < max_it && !converged && r_norm/r0_norm < div_tol) 
  {
    // Set r_n = r_n+1
    r0.assign(r1);

    // Compute A*p
    axpy_prod(A, p, Ap, true);

    // alpha = (r1,r^star)/(A*p,r^star)
    alpha = r_rstar1/ublas::inner_prod(Ap, rstar);
//    alpha = ublas::inner_prod(r1,rstar)/ublas::inner_prod(Ap, rstar);

    // s = r0 - alpha*A*p;
    noalias(s) = r0 - alpha*Ap;

    // Compute A*s
    axpy_prod(A, s, As, true);

    // omega = (A*s, s) / (A*s, A*s) 
    omega = ublas::inner_prod(As, s)/ublas::inner_prod(As, As);

    // x = x + alpha*p + omega*s 
    noalias(x) += alpha*p + omega*s;
    
    // r = s - omega*A*s
    noalias(r1) = s - omega*As;

    // Compute norm of the residual and check for convergence
    r_norm = norm_2(r1);
    if( r_norm/r0_norm < rtol || r_norm < atol)
      converged = true;
    else
    {
      r_rstar0 = r_rstar1; 
      r_rstar1 = ublas::inner_prod(r1, rstar); 
      beta  = ( r_rstar1/ r_rstar0 )*(alpha/omega);

      // p = r1 + beta*p - beta*omega*A*p
      vtemp.assign(r1+beta*p-beta*omega*Ap);      
      p.assign( vtemp );    
    }
    ++iteration;  
  }
  return iteration;
}
//-----------------------------------------------------------------------------
