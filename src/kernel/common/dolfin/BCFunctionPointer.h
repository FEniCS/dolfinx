// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __BCFUNCTION_POINTER_H
#define __BCFUNCTION_POINTER_H

#include <dolfin/constants.h>
#include <dolfin/BoundaryCondition.h>

namespace dolfin
{
  
  typedef void (*BCFunctionPointer) (BoundaryCondition&);
  typedef BCFunctionPointer bcfunction;
  
}
  
#endif
