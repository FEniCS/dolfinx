// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __P1TET_H
#define __P1TET_H

#include <dolfin/FunctionSpace.h>
#include <dolfin/ShapeFunction.h>
#include <dolfin/ExpressionFunction.h>

namespace dolfin {
  
  class P1Tet : public FunctionSpace {
  public:
    
    // Definition of the local function space
    P1Tet() : FunctionSpace(4)
    {
      // Define shape functions
      ShapeFunction v0(tet10);
      ShapeFunction v1(tet11);
      ShapeFunction v2(tet12);
      ShapeFunction v3(tet13);
      
      // Add shape functions and specify derivatives
      add(v0, -1.0, -1.0, -1.0);
      add(v1,  1.0,  0.0,  0.0);
      add(v2,  0.0,  1.0,  0.0);
      add(v3,  0.0,  0.0,  1.0);
    }
    
    // Mapping from local to global degrees of freedom
    int dof(int i, const Cell& cell) const
    {
      return cell.nodeID(i);
    }
    
    // Evalutation of degrees of freedom
    real dof(int i, const Cell& cell, const ExpressionFunction& f, real t) const
    {
      Point p = cell.coord(i);
      return f(p.x, p.y, p.z, t);
    }
    
  };
  
}

#endif
