// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __FUNCTION_SPACE_HH
#define __FUNCTION_SPACE_HH

#include <kw_constants.h>

class LocalField;
class ShapeFunction;

namespace Dolfin{ class FunctionSpace{
public:

  FunctionSpace(FiniteElement *element, int nvc);
  virtual ~FunctionSpace();

  int GetDim();
  int GetSpaceDim();
  int GetNoComponents();
  int GetCellNumber();

  real GetCoord(int node, int dim);

  int  GetNoCells();

  virtual void Update() = 0;
  
  real GetCircumRadius();

  FiniteElement* GetFiniteElement();
  ShapeFunction* GetShapeFunction(int dof);

  friend class FiniteElement;
  friend class LocalField;
  friend class TetLinFunction;
  friend class TriLinFunction;
  
protected:

  real **gradient;
  
  FiniteElement *element;
  ShapeFunction **shapefunction;

  int dim; // Scalar dimension 
  int nsd; // Number of space dimensions
  int nvc; // Number of vector components
  
}; }

#endif
