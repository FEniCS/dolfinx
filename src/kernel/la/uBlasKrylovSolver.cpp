// Copyright (C) 2006 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-05-31
// Last changed: 2006-06-23


#include <dolfin/dolfin_log.h>
#include <dolfin/DenseVector.h>
#include <dolfin/uBlasSparseMatrix.h>
#include <dolfin/uBlasKrylovSolver.h>



using namespace dolfin;

//-----------------------------------------------------------------------------
uBlasKrylovSolver::uBlasKrylovSolver() : Parametrized(), type(default_solver),
      report(false), parameters_read(false) 
{
  // Do nothing
}
//-----------------------------------------------------------------------------
uBlasKrylovSolver::uBlasKrylovSolver(Type solver) : Parametrized(), type(solver),
      report(false), parameters_read(false)
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

  // Initialise preconditioner
  P.init(A);
  
  // Reinitialise x if necessary 
  // FIXME: this erases initial guess
  x.init(b.size());

  // Read parameters if not done
  if ( !parameters_read )
    readParameters();

  // Write a message
  if ( report )
    dolfin_info("Solving linear system of size %d x %d (uBlas Krylov solver).", M, N);

  // Choose solver
  bool converged;
  uint iterations;
  switch (type)
  { 
  case gmres:
    iterations = gmresSolver(A, x, b, P, converged);
    break;
  case bicgstab:
    iterations = bicgstabSolver(A, x, b, P, converged);
    break;
  case default_solver:
    iterations = bicgstabSolver(A, x, b, P, converged);
    break;
  default:
    dolfin_warning("Requested solver type unknown. USing BiCGStab.");
    iterations = bicgstabSolver(A, x, b, P, converged);
  }

  // Check for convergence
  if( !converged )
    dolfin_warning("Krylov solver failed to converge.");
  else if ( report )
      dolfin_info("Krylov solver converged in %d iterations.", iterations);

  return iterations; 
}
//-----------------------------------------------------------------------------
dolfin::uint uBlasKrylovSolver::gmresSolver(const uBlasSparseMatrix& A, DenseVector& x, 
    const DenseVector& b, const uBlasPreconditioner& P, bool& converged) const
{
  // Get size of system
  const uint size = A.size(0);

  // Create residual vector
  ublas_vector r(size);

  // Create H matrix and h vector
  ublas_matrix_tri H(restart, restart);
  ublas_vector h(restart+1);

  // Create gamma vector
  ublas_vector gamma(restart+1);

  // Matrix containing v_k as columns.
  ublas_matrix_cmajor V(size, restart+1);

  // w vector    
  ublas_vector w(size);

  // Givens vectors
  ublas_vector c(restart), s(restart);

  // Miscellaneous storage
  real nu, temp1, temp2, r_norm = 0.0, beta0 = 0;

  converged = false;
  uint iteration = 0;
  while (iteration < max_it && !converged) 
  { 
    // Compute residual r = b -A*x
    noalias(r) = b;
    axpy_prod(A, -x, r, false); 

    // Apply preconditioner
    P.solve(r);

    // L2 norm of residual (for most recent restart)
    const real beta = norm_2(r);

    // Save intial residual (from restart 0)
    if(iteration == 0)
      beta0 = beta;

    if(beta < atol)
    { 
      converged = true;
      return iteration;
    }

    // Intialise gamma
    gamma.clear();
    gamma(0) = beta;

    // Create first column of V
    noalias(column(V, 0)) = r/beta;

    // Modified Gram-Schmidt procedure
    uint subiteration = 0;    
    uint j = 0; 
    while (subiteration < restart && iteration < max_it && !converged && r_norm/beta < div_tol) 
    {
      axpy_prod(A, column(V,j), w, true);    

      // Apply preconditioner
      P.solve(w);

      for (uint i=0; i <= j; ++i) 
      {
        h(i)= inner_prod(w, column(V,i));
        noalias(w) -= h(i)*column(V,i);
      }
      h(j+1) = norm_2(w);

      // Insert column of V (inserting v_(j+1)
      noalias(column(V,j+1)) = w/h(j+1);

      // Apply previous Givens rotations to the "new" column 
      // (this could be improved? - use more uBlas functions. 
      //  The below has been taken from old DOLFIN code.)
      for(uint i=0; i<j; ++i)
      {
        temp1 = h(i);        
        temp2 = h(i+1);        
        h(i)   = c(i)*temp1 - s(i)*temp2;      
        h(i+1) = s(i)*temp1 + c(i)*temp2 ;     
      }

      // Compute new c_i and s_i
      nu = sqrt( h(j)*h(j) + h(j+1)*h(j+1) );

      // Direct access to c & s below leads to some stranger compiler errors
      // when using vector expressions and noalias(). By using "subrange",
      // we are working with vector expressions rather than reals
      //c(j) =  h(j)/nu; 
      //s(j) = -h(j+1)/nu; 
      subrange(c, j,j+1) =  subrange(h, j,j+1)/nu;
      subrange(s, j,j+1) = -subrange(h, j+1,j+2)/nu;

      // Apply new rotation to last column   
      h(j)   = c(j)*h(j) - s(j)*h(j+1);
//      h(j+1) = 0.0;

      // Apply rotations to gamma
      temp1 = c(j)*gamma(j) - s(j)*gamma(j+1);
      gamma(j+1) = s(j)*gamma(j) + c(j)*gamma(j+1); 
      gamma(j) = temp1; 
      r_norm = fabs(gamma(j+1));

      // Add h to H matrix 
      noalias(column(H, j)) = subrange(h, 0, restart);

      // Check for convergence
      if( r_norm/beta0 < rtol || r_norm < atol )
        converged = true;

      ++iteration;
      ++subiteration;
      ++j;
    }

    // Eliminate extra rows and columns (this does not resize or copy, just addresses a range)
    ublas_matrix_range_tri Htrunc(H, ublas::range(0,subiteration), ublas::range(0,subiteration) );
    ublas_vector_range g(gamma, ublas::range(0,subiteration));

    // Solve triangular system H*g and return result in g 
    ublas::inplace_solve(Htrunc, g, ublas::upper_tag () );
    
    // x_m = x_0 + V*y
    ublas_matrix_cmajor_range v( V, ublas::range(0,V.size1()), ublas::range(0,subiteration) );
    axpy_prod(v, g, x, false);
  }  
  return iteration;
}
//-----------------------------------------------------------------------------
dolfin::uint uBlasKrylovSolver::bicgstabSolver(const uBlasSparseMatrix& A, DenseVector& x, 
    const DenseVector& b, const uBlasPreconditioner& P, bool& converged) const
{
  // Get size of system
  const uint size = A.size(0);

  // Allocate vectors
  ublas_vector r(size), rstar(size), p(size), s(size), v(size), t(size), phat(size), shat(size);
  ublas_vector vtemp(size);

  real alpha = 0.0, beta = 0.0, omega = 0.0, r_norm = 0.0; 

  // Compute residual r = b -A*x
  r.assign(b);
  axpy_prod(A, -x, r, false); 

  const real r0_norm = norm_2(r);
  if( r0_norm < atol )  
  {
    converged = true;    
    return 0;
  }  

  // Initialise p and apply preconditioner
  p.assign(r);
  phat.assign(p);
  P.solve(phat);

  // Initialise r^star?
  rstar.assign(r);
  
  // Save inner products (r0, r*) and (r1, r*) 
  real r_rstar0 = 0.0;
  real r_rstar1 = ublas::inner_prod(r,rstar);

  // Start iterations
  converged = false;
  uint iteration = 0;
  while (iteration < max_it && !converged && r_norm/r0_norm < div_tol) 
  {
    // Compute v = A*p
    axpy_prod(A, phat, v, true);

    // alpha = (r1,r^star)/(v,r^star)
    alpha = r_rstar1/ublas::inner_prod(v, rstar);

    // s = r0 - alpha*A*p;
    noalias(s) = r - alpha*v;

    shat.assign(s);
    P.solve(shat);

    // Compute t = A*s
    axpy_prod(A, shat, t, true);

    // omega = (t, s) / (t, t) 
    omega = ublas::inner_prod(t, s)/ublas::inner_prod(t, t);

    // x = x + alpha*p + omega*s 
    noalias(x) += alpha*phat + omega*shat;
    
    // r = s - omega*A*s
    noalias(r) = s - omega*t;

    // Compute norm of the residual and check for convergence
    r_norm = norm_2(r);
    if( r_norm/r0_norm < rtol || r_norm < atol)
      converged = true;
    else
    {
      r_rstar0 = r_rstar1; 
      r_rstar1 = ublas::inner_prod(r, rstar); 
      beta  = ( r_rstar1/ r_rstar0 )*(alpha/omega);

      // p = r1 + beta*p - beta*omega*A*p
      vtemp.assign(r+beta*p-beta*omega*t);      
      p.assign( vtemp );    

      phat.assign(p);
      P.solve(phat);
    }
    ++iteration;  
  }
  return iteration;
}
//-----------------------------------------------------------------------------
void uBlasKrylovSolver::readParameters()
{
  // Set tolerances and other parameters
  rtol    = get("Krylov relative tolerance");
  atol    = get("Krylov absolute tolerance");
  div_tol = get("Krylov divergence limit");
  max_it  = get("Krylov maximum iterations");
  restart = get("Krylov GMRES restart");
  report  = get("Krylov report");

  // Remember that we have read parameters
  parameters_read = true;
}
//-----------------------------------------------------------------------------
