// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include "FunctionSpace.hh"
#include "FiniteElement.hh"
#include <dolfin/Display.hh>

//-----------------------------------------------------------------------------
FunctionSpace::FunctionSpace(FiniteElement *element, int nvc)
{
  nsd = element->nsd;
  dim = 0;
  this->nvc = nvc;
  this->element = element;
  gradient = 0;
  shapefunction = 0;
}
//-----------------------------------------------------------------------------
FunctionSpace::~FunctionSpace()
{
  
}
//-----------------------------------------------------------------------------
real FunctionSpace::GetCoord(int node, int dim)
{
  return element->coord[node][dim];
}
//-----------------------------------------------------------------------------
int FunctionSpace::GetDim()
{
  return dim;
}
//-----------------------------------------------------------------------------
int FunctionSpace::GetSpaceDim()
{
  return element->GetSpaceDim();
}
//-----------------------------------------------------------------------------
int FunctionSpace::GetNoComponents()
{
  return nvc;
}
//-----------------------------------------------------------------------------
int FunctionSpace::GetCellNumber()
{
  return element->GetCellNumber();
}
//-----------------------------------------------------------------------------
int FunctionSpace::GetNoCells()
{
  return element->GetNoCells();
}
//-----------------------------------------------------------------------------
real FunctionSpace::GetCircumRadius()
{
  return element->GetCircumRadius();
}
//-----------------------------------------------------------------------------
FiniteElement* FunctionSpace::GetFiniteElement()
{
  return element;
}
//-----------------------------------------------------------------------------
ShapeFunction* FunctionSpace::GetShapeFunction(int dof)
{
  return shapefunction[dof];
}
//-----------------------------------------------------------------------------
