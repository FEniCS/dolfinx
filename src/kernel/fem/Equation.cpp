// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include "Equation.hh"
#include "GlobalField.hh"
#include "LocalField.hh"
#include "FiniteElement.hh"
#include <dolfin/Display.hh>
#include <Settings.hh>

using namespace dolfin;

//-----------------------------------------------------------------------------
Equation::Equation()
{
  dt = 0.0;
  t  = 0.0;
  h  = 0.0;
}
//-----------------------------------------------------------------------------
Equation::Equation(int nsd)
{
  dt = 0.0;
  t  = 0.0;
  h  = 0.0;
    
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

}
//-----------------------------------------------------------------------------
void Equation::updateLHS(FiniteElement *element)
{
  // Common update for LHS and RHS
  updateCommon(element);

  // Local update of LHS
  updateLHS();
}
//-----------------------------------------------------------------------------
void Equation::updateRHS(FiniteElement *element)
{
  // Common update for LHS and RHS
  updateCommon(element);

  // Local update of RHS
  updateRHS();
}
//-----------------------------------------------------------------------------
void Equation::setTime(real t)
{
  this->t = t;
}
//-----------------------------------------------------------------------------
void Equation::setTimeStep(real dt)
{
  this->dt = dt;
}
//-----------------------------------------------------------------------------
void Equation::updateCommon(FiniteElement *element)
{
  // Update cell size
  h = 2.0 * element->GetCircumRadius();
}
//----------------------------------------------------------------------------
