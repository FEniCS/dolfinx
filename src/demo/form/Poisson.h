// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __POISSON_H
#define __POISSON_H

#include <dolfin/PDE.h>

using namespace dolfin;
  
class Poisson : public PDE
{
public:
  
  Poisson() : PDE(3) {}
  
  real lhs(const ShapeFunction& u, const ShapeFunction& v)
  {
    return (grad(u),grad(v)) * dx;
  }
  
};

#endif
