// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __LOCALFIELD_HH
#define __LOCALFIELD_HH

#include <kw_constants.h>
#include "FiniteElement.hh"
#include "GlobalField.hh"

class FunctionSpace;
class ShapeFunction;

namespace Dolfin{ class LocalField {
  
public:

  LocalField();
  ~LocalField();

  void AttachGlobalField(GlobalField *globalfield, int component);
  void Update(FiniteElement *element, real t);

  void Add(LocalField &lf);
  void Mult(real a);
  void SetToConstant(real a);
  void Mean(LocalField &v, LocalField &w);
  
  real dx;
  real dy;
  real dz;
  
  int  GetDim() const;
  int  GetCellNumber() const;
  real GetCoord(int node, int dim) const;
  real GetDofValue(int i) const;
  real GetMeanValue() const;

  void Display();
  
  FiniteElement*       GetFiniteElement() const;
  FunctionSpace*       GetFunctionSpace();
  const ShapeFunction* GetShapeFunction(int i) const; 

  // Operators

  void operator = (LocalField &lf);
  
  real operator* (const LocalField &v) const;
  real operator* (const ShapeFunction &v) const;
  real operator* (real a) const;

  friend real operator* (real a, LocalField &v);
  friend real operator* (const ShapeFunction &v, const LocalField &w);
  
protected:

  void Resize(FunctionSpace *functionspace);
  void ComputeGradient();
  
  real *dof;
  
  int dim;
  int nsd;
  int component;

  GlobalField *globalfield;
  FunctionSpace *functionspace;
  ShapeFunction **shapefunction;
  
}; }

#endif
