// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __VFUNCTION_POINTER_H
#define __VFUNCTION_POINTER_H

#include <dolfin/constants.h>

namespace dolfin
{
  
  typedef real (*VFunctionPointer) (real, real, real, real, int);
  typedef VFunctionPointer vfunction;
  
}
  
#endif
