// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __EQUATION_SYSTEM_HH
#define __EQUATION_SYSTEM_HH

#include "Equation.hh"
#include <Display.hh>

class EquationSystem: public Equation{

public:
  
  EquationSystem(int noeq, int nsd);
  ~EquationSystem(){};
  
  virtual real IntegrateLHS(ShapeFunction *u, ShapeFunction *v) = 0;
  virtual real IntegrateRHS(ShapeFunction *v) = 0;

  // Implementation of integrators for base class
  real IntegrateLHS(ShapeFunction &u, ShapeFunction &v){
    display->InternalError("EquationSystem::IntegrateLHS()",
			   "Using EquationSystem for equation with only one component.");
    return 0.0;
  }
  real IntegrateRHS(ShapeFunction &v){
    display->InternalError("EquationSystem::IntegrateRHS()",
			   "Using EquationSystem for equation with only one component.");
    return 0.0;
  }

};

#endif
