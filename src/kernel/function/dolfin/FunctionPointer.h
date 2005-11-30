// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-11-28
// Last changed: 2005-11-28

#ifndef __FUNCTION_POINTER_H
#define __FUNCTION_POINTER_H

#include <dolfin/Point.h>

namespace dolfin
{

  /// Function pointer type for user defined functions
  typedef real(*FunctionPointer)(const Point& p, unsigned int);

 }

#endif
