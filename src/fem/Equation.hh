// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __EQUATION_HH
#define __EQUATION_HH

#include <kw_constants.h>
#include "LocalField.hh"
#include "GlobalField.hh"
#include "ShapeFunction.hh"

class GlobalField;

///
namespace Dolfin{ class Equation {

public:
  
  Equation(int nsd);
  ~Equation();

  virtual real IntegrateLHS(ShapeFunction &u, ShapeFunction &v) = 0;
  virtual real IntegrateRHS(ShapeFunction &v) = 0;

  void UpdateLHS(FiniteElement *element);
  void UpdateRHS(FiniteElement *element);
  void AttachField(int i, GlobalField *globalfield, int component = 0);

  void SetTime     (real t);
  void SetTimeStep (real dt);
  
  int GetNoEq();    
  int GetStartVectorComponent();

  friend class Discretiser;
  
protected:
  
  void AllocateFields(int no_fields);
  void UpdateCommon(FiniteElement *element);

  virtual void UpdateLHS() { };
  virtual void UpdateRHS() { };
  
  LocalField **field;
  int no_fields;

  int start_vector_component;

  int nsd;   // number of space dimensions
  int no_eq; // number of equations

  real dt;
  real t;  
  real h;

  void (*update)();
  
}; }

#endif
