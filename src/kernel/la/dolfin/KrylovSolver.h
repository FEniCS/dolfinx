// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __KRYLOV_SOLVER_H
#define __KRYLOV_SOLVER_H

#include <dolfin/constants.h>

namespace dolfin {

  class Matrix;
  class Vector;

  /// Krylov solver, including conjugate gradient (CG) and GMRES
  class KrylovSolver{
  public:
    
    enum Method { GMRES, CG };
    enum Preconditioner { RICHARDSON, JACOBI, GAUSS_SEIDEL, SOR, NONE };
    
    KrylovSolver(Method method = GMRES);
    
    void solve(const Matrix& A, Vector& x, const Vector& b);

    void setMethod(Method method);
    
  private:
    
    void solveCG    (const Matrix& A, Vector& x, const Vector& b);
    void solveGMRES (const Matrix& A, Vector& x, const Vector& b);
    
    int restartedGMRES(const Matrix& A, Vector& x, const Vector& b, int k_max);
    
    void solvePxv (const Matrix& A, Vector& x, Matrix& v, int k);
    void solvePxu (const Matrix& A, Vector& x, Vector& u);
    void applyPxu (const Matrix& A, Vector& x, Vector& u);
    
    bool reorthog (const Matrix& A, Matrix& v, int k);
    
    real residual (const Matrix& A, Vector& x, const Vector& b, Vector& r);
    real residual (const Matrix& A, Vector& x, const Vector& b);
    
    Method method;
    Preconditioner pc;
    
    real tol;
    int no_pc_sweeps;
  };
  
}

#endif
