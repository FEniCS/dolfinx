// Copyright (C) 2008 Kent-Andre Mardal.
// Licensed under the GNU LGPL Version 2.1.
//
// Last changed: 2008-05-16

#ifdef HAS_TRILINOS

#include <dolfin/log/dolfin_log.h>
#include "EpetraUserPreconditioner.h"

using namespace dolfin;
//-----------------------------------------------------------------------------
void EpetraUserPreconditioner::set_type(std::string type)
{
  this->type = type;
}
//-----------------------------------------------------------------------------
void EpetraUserPreconditioner::init(const EpetraMatrix& a) 
{
  error("EpetraUserPreconditioner::init: not implemented ");
}
//-----------------------------------------------------------------------------
void EpetraUserPreconditioner::solve(EpetraVector& x, const EpetraVector& b) 
{
  error("EpetraUserPreconditioner::solve: not implemented ");
}
//-----------------------------------------------------------------------------
#endif


