// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __FGMRES_H
#define __FGMRES_H

#include <dolfin/constants.h>
#include <dolfin/Preconditioner.h>
#include <dolfin/LinearSolver.h>

namespace dolfin
{
  
  class Matrix;
  class Vector;
  
  /// This is a flexible GMRES with right preconditioning,
  /// due to Y. Saad, SIAM J. Sci. Comput., 14 (1993), 461-469.
  
  class FGMRES : public Preconditioner, public LinearSolver
    {
    public:
	
    /// Creates a FGMRES solver for a given matrix. Return an error 
    /// message if convergence is not obtained.
    FGMRES(const Matrix& A, unsigned int restarts, unsigned int maxiter, 
	   real tol, Preconditioner& pc);

    /// Creates a FGMRES solver/preconditioner for a given matrix.
    /// Does not return an error message if convergence is not 
    /// obtained. Recommended for preconditioning.
    FGMRES(const Matrix& A, unsigned int restarts, real tol, 
	   Preconditioner& pc);

    /// Destructor
    ~FGMRES();
	  
    /// Solve linear system Ax = b for a given right-hand side b
    void solve(Vector& x, const Vector& b);
	    
    /// Solve linear system Ax = b for a given right-hand side b
    static void solve(const Matrix& A, Vector& x, const Vector& b,
		      unsigned int restarts, unsigned int maxiter, real tol,
		      Preconditioner& pc);
    	    
    private:

    // Main FGMRES iterator.
    unsigned int iterator(Vector& x, const Vector& b, Vector& r);

    // Reorthogonalize.
    bool reorthog(Matrix& v, Vector &x, int k);
	      
    // The matrix.
    const Matrix& A;

    // Maximum number of restarts.
    unsigned int restarts;

    // Number of iterations per restart.
    unsigned int maxiter;

    // Tolerance
    real tol;  

    // The preconditioner.
    Preconditioner& pc;

    // Switch, solve to convergence. 
    bool solve2convergence; 

    // Norm of right hand side.
    real bnorm;
      
   };
}

#endif
