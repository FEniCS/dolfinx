// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include "Equation.hh"
#include "GlobalField.hh"
#include "LocalField.hh"
#include "FiniteElement.hh"
#include <dolfin/Display.hh>
#include <Settings.hh>

using namespace Dolfin;

//-----------------------------------------------------------------------------
Equation::Equation(int nsd)
{
  this->nsd = nsd;
  no_eq     = 1;

  field = 0;
  no_fields = 0;

  dt = 0.0;
  t  = 0.0;
  h  = 0.0;
    
  start_vector_component = 0;

  update = 0;
  
  // Check that the dimension matches
  int space_dimension;
  settings->Get("space dimension",&space_dimension);
  if ( space_dimension != nsd )
	 display->Error("Specified dimension (%d) does not equation dimension (%d)",
						 space_dimension,nsd);
}
//-----------------------------------------------------------------------------
Equation::~Equation()
{
  if ( field )
    delete [] field;
  field = 0;
  no_fields = 0;
}
//-----------------------------------------------------------------------------
void Equation::UpdateLHS(FiniteElement *element)
{
  // Common update for LHS and RHS
  UpdateCommon(element);

  // Local update of LHS
  UpdateLHS();
}
//-----------------------------------------------------------------------------
void Equation::UpdateRHS(FiniteElement *element)
{
  // Common update for LHS and RHS
  UpdateCommon(element);

  // Local update of RHS
  UpdateRHS();
}
//-----------------------------------------------------------------------------
void Equation::AttachField(int i, GlobalField *globalfield, int component = 0)
{  
  if ( (i < 0) || (i >= no_fields ) )
	 display->Error("Illegal index for field: %d",i);

  if ( !field[i] )
	 display->Error("Local field %d is not specified.",i);
  
  field[i]->AttachGlobalField(globalfield,component);
}
//-----------------------------------------------------------------------------
void Equation::SetTime(real t)
{
  this->t = t;
}
//-----------------------------------------------------------------------------
void Equation::SetTimeStep(real dt)
{
  this->dt = dt;
}
//-----------------------------------------------------------------------------
int Equation::GetStartVectorComponent()
{
  return start_vector_component;
}
//----------------------------------------------------------------------------
int Equation::GetNoEq()
{
  return no_eq;
}
//----------------------------------------------------------------------------
void Equation::AllocateFields(int no_fields)
{
  field = new (LocalField *)[no_fields];
  this->no_fields = no_fields;

  for (int i=0;i<no_fields;i++)
	 field[0] = 0;
}
//----------------------------------------------------------------------------
void Equation::UpdateCommon(FiniteElement *element)
{
  // Update all local fields
  for (int i=0;i<no_fields;i++){
    if ( !field[i] )
      display->Error("Local field %d is not specified.",i);
    field[i]->Update(element,t);
  }
  
  // Update cell size
  h = 2.0 * element->GetCircumRadius();
}
//----------------------------------------------------------------------------
