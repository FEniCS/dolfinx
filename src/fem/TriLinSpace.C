// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/Display.hh>
#include <utils.h>
#include <kw_constants.h>
#include "TriLinSpace.hh"
#include "TriLinFunction.hh"
#include "LocalField.hh"
#include "FiniteElement.hh"
#include <dolfin/Grid.hh>

//-----------------------------------------------------------------------------
TriLinSpace::TriLinSpace(FiniteElement *element,int nvc) : FunctionSpace(element,nvc)
{
  dim = 3;

  shapefunction = new (ShapeFunction *)[dim];
 
  for (int i=0;i<dim;i++)
    shapefunction[i] = new TriLinFunction[nvc](this,i);
  
  gradient = new(real *)[dim];
  for (int i = 0; i < dim; i++)
    gradient[i] = new real[nsd];
}
//-----------------------------------------------------------------------------
TriLinSpace::~TriLinSpace()
{
  for (int i=0;i<dim;i++)
    delete [] shapefunction[i];
  delete [] shapefunction;
  shapefunction = 0;
  
  for (int i=0;i<dim;i++)
    delete [] gradient[i];
  delete [] gradient;
  gradient = 0;
}
//-----------------------------------------------------------------------------
void TriLinSpace::Update()
{
  Cell *c;
  c = element->grid->GetCell(element->GetCellNumber());

  if (c->GetSize() != dim)
    display->Error("Function space not compatible with cell geometry (triangles)");
	 
  g2x = element->coord[2][1] - element->coord[0][1];
  g2y = element->coord[0][0] - element->coord[2][0];
  g3x = element->coord[0][1] - element->coord[1][1];
  g3y = element->coord[1][0] - element->coord[0][0];

  det = g3y*g2x - g3x*g2y;  
  d   = 1.0/det;
  
  g2x *= d;
  g2y *= d;
  g3x *= d;
  g3y *= d;
  
  g1x = - g2x - g3x;
  g1y = - g2y - g3y;

  gradient[0][0] = g1x;
  gradient[0][1] = g1y;
  gradient[1][0] = g2x;
  gradient[1][1] = g2y;
  gradient[2][0] = g3x;
  gradient[2][1] = g3y;
}
//-----------------------------------------------------------------------------
real TriLinSpace::IntShapeFunction(int m1)
{
  return real(2*factorial(m1)) / real(factorial(m1+2));
}
//-----------------------------------------------------------------------------
real TriLinSpace::IntShapeFunction(int m1, int m2)
{
  return real(2*factorial(m1)*factorial(m2)) / real(factorial(m1+m2+2));
}
//-----------------------------------------------------------------------------
real TriLinSpace::IntShapeFunction(int m1, int m2, int m3)
 {
  return real(2*factorial(m1)*factorial(m2)*factorial(m3)) /
         real(factorial(m1+m2+m3+2));
}
//-----------------------------------------------------------------------------
