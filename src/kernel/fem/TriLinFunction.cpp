// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include "TriLinFunction.hh"
#include "TriLinSpace.hh"
#include "ShapeFunction.hh"
#include "FiniteElement.hh"
#include "LocalField.hh"
#include <dolfin/Display.hh> 

//-----------------------------------------------------------------------------
TriLinFunction::TriLinFunction(FunctionSpace *functionspace, int dof) :
  ShapeFunction(functionspace,dof)
{

}
//-----------------------------------------------------------------------------
TriLinFunction::~TriLinFunction()
{

}
//-----------------------------------------------------------------------------
void TriLinFunction::barycentric(real *point, real *bcoord)
{
  point[0]-=functionspace->GetCoord(0,0);
  point[1]-=functionspace->GetCoord(0,1);
  point[2]-=functionspace->GetCoord(0,2);

  bcoord[1] = functionspace->gradient[1][0]*point[0] + 
              functionspace->gradient[1][1]*point[1] + 
              functionspace->gradient[1][2]*point[2]; 
  bcoord[2] = functionspace->gradient[2][0]*point[0] + 
              functionspace->gradient[2][1]*point[1] + 
              functionspace->gradient[2][2]*point[2];
  bcoord[3] = functionspace->gradient[3][0]*point[0] + 
              functionspace->gradient[3][1]*point[1] + 
              functionspace->gradient[3][2]*point[2];    
  bcoord[0] = 1 - bcoord[1] - bcoord[2] - bcoord[3];
}
//-----------------------------------------------------------------------------
real TriLinFunction::operator* (const ShapeFunction &v) const
{
  if ( !active || !v.active ) return 0.0;

  if ( dof == v.dof )
	 return DOLFIN_1_6;
  else
	 return DOLFIN_1_12;
}
//-----------------------------------------------------------------------------
real TriLinFunction::operator* (real a) const
{
  if ( !active ) return 0.0;

  return a*DOLFIN_1_3;
}
//-----------------------------------------------------------------------------
