// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __KRYLOV_SOLVER_H
#define __KRYLOV_SOLVER_H

#include <dolfin/constants.h>

namespace dolfin {

  class Matrix;
  class DenseMatrix;
  class Vector;

  /// Krylov solver, including conjugate gradient (CG) and GMRES
  class KrylovSolver{
  public:
    
    enum Method { GMRES, CG };
    enum Preconditioner { RICHARDSON, JACOBI, GAUSS_SEIDEL, SOR, NONE };
    
    KrylovSolver(Method method = GMRES);
    
    void solve(Matrix& A, Vector& x, Vector& b);

    void setMethod(Method method);
    
  private:
    
    void solveCG(Matrix &A, Vector &x, Vector &b);
    void solveGMRES(Matrix &A, Vector &x, Vector &b);
    
    int restartedGMRES(Matrix &A, Vector &x, Vector &b, int k_max);
    
    void solvePxv(Matrix &A, Vector &x, DenseMatrix &v, int k);
    void solvePxu(Matrix &A, Vector &x, Vector &u);
    void applyPxu(Matrix &A, Vector &x, Vector &u);
    
    bool reOrthogonalize(Matrix &A, DenseMatrix &v, int k);
    
    real getResidual(Matrix &A, Vector &x, Vector &b, Vector &r);
    real getResidual(Matrix &A, Vector &x, Vector &b);
    
    Method method;
    Preconditioner pc;
    
    real tol;
    int no_pc_sweeps;
  };
  
}

#endif
