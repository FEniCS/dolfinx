// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include "ShapeFunction.hh"
#include "FunctionSpace.hh"
#include <Settings.hh>
#include <Display.hh>

//-----------------------------------------------------------------------------
ShapeFunction::ShapeFunction(FunctionSpace *functionspace, int dof)
{
  this->functionspace = functionspace;
  this->dof = dof;
  active = true;

  dx = 0.0;
  dy = 0.0;
  dz = 0.0;
}
//-----------------------------------------------------------------------------
ShapeFunction::~ShapeFunction()
{

}
//-----------------------------------------------------------------------------
void ShapeFunction::SetDof(int dof)
{
  this->dof = dof;
}
//-----------------------------------------------------------------------------
void ShapeFunction::Active(bool active)
{
  this->active = active;
}
//-----------------------------------------------------------------------------
bool ShapeFunction::Active() const
{
  return active;
}
//-----------------------------------------------------------------------------
void ShapeFunction::Display()
{
  display->Message(0,"ShapeFunction: dof = %d active = %d",dof,active);
}
//-----------------------------------------------------------------------------
int ShapeFunction::GetDof() const
{
  return dof;
}
//-----------------------------------------------------------------------------
int ShapeFunction::GetDim()
{
  return functionspace->GetDim();
}
//-----------------------------------------------------------------------------
int ShapeFunction::GetCellNumber()
{
  return functionspace->GetCellNumber();
}
//-----------------------------------------------------------------------------
int ShapeFunction::GetNoCells()
{
  return functionspace->GetNoCells();
}
//-----------------------------------------------------------------------------
real ShapeFunction::GetCircumRadius()
{
  return functionspace->GetCircumRadius();
}
//-----------------------------------------------------------------------------
real ShapeFunction::GetCoord(int node, int dim)
{
  return functionspace->GetCoord(node,dim);
}
//-----------------------------------------------------------------------------
FiniteElement* ShapeFunction::GetFiniteElement()
{
  return functionspace->GetFiniteElement();
}
//-----------------------------------------------------------------------------
real operator* (real a, ShapeFunction &v)
{
  // This makes sure that a * ShapeFunction is commutative

  return ( v*a );
}
//-----------------------------------------------------------------------------
