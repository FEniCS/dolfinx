// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Thomas Svedberg, 2004.

#ifndef __NEW_KRYLOV_SOLVER_H
#define __NEW_KRYLOV_SOLVER_H

#include <dolfin/constants.h>

namespace dolfin {

 class Matrix;
 class Vector;
  
 /// Wrapper for PETSc Krylov solvers.
 ///
 /// Only a placeholder for now

 class KrylovSolver
 {
 public:
   
   enum Method { GMRES, CG, BiCGSTAB };
   enum Preconditioner { RICHARDSON, JACOBI, GAUSS_SEIDEL, SOR, NONE };
   
   KrylovSolver(Method method = GMRES);
   
   void solve(const NewMatrix &A, NewVector &x, const NewVector &b);

   void setMethod(Method method);
   void setPreconditioner(Preconditioner pc);
   
 private:
   
 };
  
}

#endif
