// Copyright (C) 2006 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg 2006.
//
// First added:  2006-05-31
// Last changed: 2006-07-03

#include <dolfin/dolfin_log.h>
#include <dolfin/DenseVector.h>
#include <dolfin/uBlasKrylovMatrix.h>
#include <dolfin/uBlasKrylovSolver.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
uBlasKrylovSolver::uBlasKrylovSolver()
  : Parametrized(),
    type(default_solver), report(false), parameters_read(false) 
{
  // Do nothing
}
//-----------------------------------------------------------------------------
uBlasKrylovSolver::uBlasKrylovSolver(Type solver)
  : Parametrized(),
    type(solver), report(false), parameters_read(false)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
uBlasKrylovSolver::~uBlasKrylovSolver()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
dolfin::uint uBlasKrylovSolver::solve(const uBlasKrylovMatrix& A,
				      DenseVector& x, const DenseVector& b)
{
  // Check dimensions
  uint M = A.size(0);
  uint N = A.size(1);
  if ( N != b.size() )
    dolfin_error("Non-matching dimensions for linear system.");

  // Initialise preconditioner
  // FIXME: Preconditioner temporarily disabled
  //P.init(A);
  
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
    dolfin_warning("Requested solver type unknown. Using BiCGStab.");
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
dolfin::uint uBlasKrylovSolver::gmresSolver(const uBlasKrylovMatrix& A,
					    DenseVector& x, 
					    const DenseVector& b,
					    const uBlasPreconditioner& P,
					    bool& converged) const
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
    //noalias(r) = b;
    //axpy_prod(A, -x, r, false); 
    A.mult(x, static_cast<DenseVector&>(r));
    r *= -1.0;
    r += b;

    // Apply preconditioner
    // FIXME: Preconditioner temporarily disabled
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
      // Compute product w = A*V_j (use r for temporary storage)
      //axpy_prod(A, column(V, j), w, true);
      noalias(r) = column(V, j);
      A.mult(static_cast<DenseVector&>(r), static_cast<DenseVector&>(w));

      // Apply preconditioner
      // FIXME: Preconditioner temporarily disabled
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
dolfin::uint uBlasKrylovSolver::bicgstabSolver(const uBlasKrylovMatrix& A,
					       DenseVector& x,
					       const DenseVector& b,
					       const uBlasPreconditioner& P,
					       bool& converged) const
{
  // Get size of system
  const uint size = A.size(0);

  // Allocate vectors
  ublas_vector r(size), rstar(size), p(size), s(size), v(size), t(size), y(size), z(size);

  real alpha = 1.0, beta = 0.0, omega = 1.0, r_norm = 0.0; 
  real rho_old = 1.0, rho = 1.0;

  // Compute residual r = b -A*x
  //r.assign(b);
  //axpy_prod(A, -x, r, false);
  A.mult(x, static_cast<DenseVector&>(r));
  r *= -1.0;
  r += b;

  const real r0_norm = norm_2(r);
  if( r0_norm < atol )  
  {
    converged = true;    
    return 0;
  }  

  // Initialise r^star, v and p
  rstar.assign(r);
  v.clear();
  p.clear();

  // Apply preconditioner to r^start. This is a trick to avoid problems in which 
  // (r^start, r) = 0  after the first iteration (such as PDE's with homogeneous 
  // Neumann bc's and no forcing/source term.
  // FIXME: Preconditioner temporarily disabled
  P.solve(rstar);

  // Right-preconditioned Bi-CGSTAB

  // Start iterations
  converged = false;
  uint iteration = 0;
  while (iteration < max_it && !converged && r_norm/r0_norm < div_tol) 
  {
    // Set rho_n = rho_n+1
    rho_old = rho; 

    // Compute new rho
    rho = ublas::inner_prod(r, rstar); 
    if( fabs(rho) < 1e-25 )
      dolfin_error1("BiCGStab breakdown. rho = %g", rho);

    beta = (rho/rho_old)*(alpha/omega);

    // p = r1 + beta*p - beta*omega*A*p
    p *= beta;
    noalias(p) += r - beta*omega*v;

    // My = p
    y.assign(p);
    // FIXME: Preconditioner temporarily disabled
    P.solve(y);

    // v = A*y
    //axpy_prod(A, y, v, true);
    A.mult(static_cast<DenseVector&>(y), static_cast<DenseVector&>(v));

    // alpha = (r, rstart) / (v, rstar)
    alpha = rho/ublas::inner_prod(v, rstar);

    // s = r - alpha*v
    noalias(s) = r - alpha*v;
    
    // Mz = s
    z.assign(s);
    P.solve(z);

    // t = A*z
    //axpy_prod(A, z, t, true);
    A.mult(static_cast<DenseVector&>(z), static_cast<DenseVector&>(t));

    // omega = (t, s) / (t,t)
    omega = ublas::inner_prod(t, s)/ublas::inner_prod(t, t);

    // x = x + alpha*p + omega*s
    noalias(x) += alpha*y + omega*z;

    // r = s - omega*t
    noalias(r) = s - omega*t;

    // Compute norm of the residual and check for convergence
    r_norm = norm_2(r);
    if( r_norm/r0_norm < rtol || r_norm < atol)
      converged = true;

    ++iteration;  
  }


/*
  // Left-conditioned Bi-CGSTAB
  P.solve(r);
  r0_norm = norm_2(r);
  rstar.assign(r);
  v.clear();
  p.clear();

  // Start iterations
  converged = false;
  uint iteration = 0;
  while (iteration < max_it && !converged && r_norm/r0_norm < div_tol) 
  {
    // Set rho_n = rho_n+1
    rho_old = rho; 

    // Compute new rho
    rho = ublas::inner_prod(r, rstar); 
    if( fabs(rho) < 1e-25 )
      dolfin_error1("BiCGStab breakdown (2). rho = %g", rho);

    beta = (rho/rho_old)*(alpha/omega);

    // p = r1 + beta*p - beta*omega*A*p
    p *= beta;
    noalias(p) += r - beta*omega*v;

    // v = A*p
    axpy_prod(A, p, v, true);
    P.solve(v);

    // alpha = (r, rstart) / (v, rstar)
    alpha = rho/ublas::inner_prod(v, rstar);

    // s = r - alpha*v
    noalias(s) = r - alpha*v;
    
    // t = A*s
    axpy_prod(A, s, t, true);
    P.solve(t);

    // omega = (t, s) / (t,t)
    omega = ublas::inner_prod(t, s)/ublas::inner_prod(t, t);

    // x = x + alpha*p + omega*s
    noalias(x) += alpha*p + omega*s;

    // r = s - omega*t
    noalias(r) = s - omega*t;

    // Compute norm of the residual and check for convergence
    r_norm = norm_2(r);
    if( r_norm/r0_norm < rtol || r_norm < atol)
      converged = true;

    ++iteration;  
  }
*/
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
