// Copyright (C) 2008 Kent-Andre Mardal.
// Licensed under the GNU LGPL Version 2.1.
//
// Last changed: 2008-05-16

#ifdef HAS_TRILINOS

#include "EpetraPreconditioner.h"

using namespace dolfin;
//-----------------------------------------------------------------------------
void EpetraPreconditioner::setType(PreconditionerType type) {
  prec_type = type; 
}
//-----------------------------------------------------------------------------
void EpetraPreconditioner::init(const EpetraMatrix& a) { 
  error("EpetraPreconditioner::init : not implemented "); 
}
//-----------------------------------------------------------------------------
void EpetraPreconditioner::solve(EpetraVector& x, const EpetraVector& b) { 
  error("EpetraPreconditioner::solve: not implemented "); 
}
//-----------------------------------------------------------------------------
#endif 


