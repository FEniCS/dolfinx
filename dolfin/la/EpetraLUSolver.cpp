// Copyright (C) 2008 Kent-Andre Mardal.
// Licensed under the GNU LGPL Version 2.1.
//
// Last changed: 2008-05-16

#ifdef HAS_TRILINOS

#include "EpetraMatrix.h"
#include "EpetraVector.h"
#include "EpetraLUSolver.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
EpetraLUSolver::EpetraLUSolver() {
}
//-----------------------------------------------------------------------------
EpetraLUSolver::~EpetraLUSolver() {
}
//-----------------------------------------------------------------------------
dolfin::uint EpetraLUSolver::solve(const EpetraMatrix&A, EpetraVector& x, const EpetraVector& b){
  error("EpetraLUSolver::solve not implemented"); 
  return 0; 
}
//-----------------------------------------------------------------------------
void EpetraLUSolver::disp() const {
  error("EpetraLUSolver::disp not implemented"); 
}




#endif 

