// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __DIRECT_SOLVER_HH
#define __DIRECT_SOLVER_HH

class DenseMatrix;
class SparseMatrix;
class Vector;

class DirectSolver{
public:

  DirectSolver(){}
  ~DirectSolver(){}

  void LU    (DenseMatrix *LU);
  void Solve (DenseMatrix *LU, Vector *x, Vector *b);
  void Solve (SparseMatrix *A, Vector *x, Vector *b);
  
private:
  
  
  
  
};

#endif
