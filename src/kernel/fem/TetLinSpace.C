// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/Display.hh>
#include <utils.h>
#include <kw_constants.h>
#include "TetLinSpace.hh"
#include "TetLinFunction.hh"
#include "LocalField.hh"
#include "FiniteElement.hh"
#include <dolfin/Grid.hh>

//-----------------------------------------------------------------------------
TetLinSpace::TetLinSpace(FiniteElement *element,int nvc) : FunctionSpace(element,nvc)
{
  dim = 4;

  shapefunction = new (ShapeFunction *)[dim];
 
  for (int i=0;i<dim;i++)
    shapefunction[i] = new TetLinFunction[nvc](this,i);
  
  gradient = new(real *)[dim];
  for (int i = 0; i < dim; i++)
    gradient[i] = new real[nsd];
}
//-----------------------------------------------------------------------------
TetLinSpace::~TetLinSpace()
{
  for (int i=0;i<dim;i++)
    delete [] shapefunction[i];
  delete [] shapefunction;
  shapefunction = 0;
  
  for (int i = 0; i < dim; i++)
    delete [] gradient[i];
  delete [] gradient;
  gradient = 0;
}
//-----------------------------------------------------------------------------
void TetLinSpace::Update()
{
  Cell *c;
  c = element->grid->GetCell(element->GetCellNumber());
  
  if (c->GetSize() != dim)
    display->Error("Shape function not compatible with cell geometry (tetrahedrons)");
  
  // This is only valid for linear tetrahedrons
  
  j11 = element->coord[1][0] - element->coord[0][0];
  j12 = element->coord[1][1] - element->coord[0][1];
  j13 = element->coord[1][2] - element->coord[0][2];
  j21 = element->coord[2][0] - element->coord[0][0];
  j22 = element->coord[2][1] - element->coord[0][1];
  j23 = element->coord[2][2] - element->coord[0][2];
  j31 = element->coord[3][0] - element->coord[0][0];
  j32 = element->coord[3][1] - element->coord[0][1];
  j33 = element->coord[3][2] - element->coord[0][2];
  
  gradient[1][0] = j22 * j33 - j23 * j32;
  gradient[2][0] = j13 * j32 - j12 * j33;
  gradient[3][0] = j12 * j23 - j13 * j22;
  gradient[1][1] = j23 * j31 - j21 * j33;
  gradient[2][1] = j11 * j33 - j13 * j31;
  gradient[3][1] = j13 * j21 - j11 * j23;
  gradient[1][2] = j21 * j32 - j22 * j31;
  gradient[2][2] = j12 * j31 - j11 * j32;
  gradient[3][2] = j11 * j22 - j12 * j21;
  
  det = j11 * gradient[1][0] + 
        j12 * gradient[1][1] + 
        j13 * gradient[1][2];
  
  d = 1.0 / det;

  gradient[1][0] *= d;
  gradient[2][0] *= d;
  gradient[3][0] *= d;
  
  gradient[1][1] *= d;
  gradient[2][1] *= d;
  gradient[3][1] *= d;

  gradient[1][2] *= d;
  gradient[2][2] *= d;
  gradient[3][2] *= d;

  gradient[0][0] = - ( gradient[1][0] + gradient[2][0] + gradient[3][0] );
  gradient[0][1] = - ( gradient[1][1] + gradient[2][1] + gradient[3][1] );
  gradient[0][2] = - ( gradient[1][2] + gradient[2][2] + gradient[3][2] );

  //for (int i=0;i<4;i++)
  //	 for (int j=0;j<3;j++)
  //		display->Message(0,"d = %f",gradient[i][j]);
}
//-----------------------------------------------------------------------------
real TetLinSpace::IntShapeFunction(int m1)
{
  return real(6*factorial(m1)) / real(factorial(m1+3));
}
//-----------------------------------------------------------------------------
real TetLinSpace::IntShapeFunction(int m1, int m2)
{
  return real(6*factorial(m1)*factorial(m2)) / real(factorial(m1+m2+3));
}
//-----------------------------------------------------------------------------
real TetLinSpace::IntShapeFunction(int m1, int m2, int m3)
{
  return real(6*factorial(m1)*factorial(m2)*factorial(m3)) /
         real(factorial(m1+m2+m3+3));
}
//-----------------------------------------------------------------------------
real TetLinSpace::IntShapeFunction(int m1, int m2, int m3, int m4)
{
  return real(6*factorial(m1)*factorial(m2)*factorial(m3)*factorial(m4)) /
         real(factorial(m1+m2+m3+m4+3));
}
//-----------------------------------------------------------------------------
