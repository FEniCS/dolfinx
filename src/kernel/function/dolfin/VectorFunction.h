// Copyright (C) 2004 Johan Hoffman, Johan Jansson and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __VECTORFUNCTION_H
#define __VECTORFUNCTION_H

#include <dolfin/Variable.h>
#include <dolfin/NewArray.h>
#include <dolfin/Function.h>
#include <dolfin/Point.h>

namespace dolfin {

class VectorFunction : public Variable, public NewArray<Function>
{
public:
    
  /// Constructor
  VectorFunction() : NewArray<Function>() {}

  /// Constructor
  VectorFunction(unsigned int n) : NewArray<Function>(n) {}
  
  NewArray<real> operator()(const Point& p, real t = 0.0);
  
};


}

#endif
