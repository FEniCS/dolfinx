// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __OLDPOISSON_H
#define __OLDPOISSON_H

#include <dolfin/PDE.h>

using namespace dolfin;
  
class OldPoisson : public PDE
{
public:
  
  OldPoisson() : PDE(3) {}
  
  real lhs(const ShapeFunction& u, const ShapeFunction& v)
  {
    return (grad(u),grad(v)) * dx;
  }
  
};

#endif
