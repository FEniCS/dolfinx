// Copyright (C) 2005-2007 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-11-28
// Last changed: 2007-04-05

#ifndef __FUNCTION_POINTER_H
#define __FUNCTION_POINTER_H

#include <ufc.h>
#include <dolfin/constants.h>

namespace dolfin
{

  /// Function pointer type for user defined functions
  typedef void(*FunctionPointer)(real* values, const real* coordinates);

}

#endif
