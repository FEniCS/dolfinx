// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __FUNCTION_POINTER_H
#define __FUNCTION_POINTER_H

#include <dolfin/constants.h>

namespace dolfin
{
  
  typedef real (*FunctionPointer) (real, real, real, real);
  typedef FunctionPointer function;

}
  
#endif
