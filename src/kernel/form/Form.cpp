// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg, 2005.

#include <dolfin/dolfin_log.h>
#include <dolfin/dolfin_math.h>
#include <dolfin/Cell.h>
#include <dolfin/Point.h>
#include <dolfin/NewFunction.h>
#include <dolfin/NewFiniteElement.h>
#include <dolfin/Form.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Form::Form(const NewFiniteElement& element, uint ncoeff)
  : element(element), c(0), ncoeff(0)
{
  // Reset data
  det = 0.0;
  
  f00 = 0.0; f01 = 0.0; f02 = 0.0;
  f10 = 0.0; f11 = 0.0; f12 = 0.0;
  f20 = 0.0; f21 = 0.0; f22 = 0.0;

  g00 = 0.0; g01 = 0.0; g02 = 0.0;
  g10 = 0.0; g11 = 0.0; g12 = 0.0;
  g20 = 0.0; g21 = 0.0; g22 = 0.0;

  // Initialize coefficients
  if ( ncoeff > 0 )
  {
    c = new real* [ncoeff];
    for (uint i = 0; i < ncoeff; i++)
    {
      c[i] = new real[element.spacedim()];
      for (uint j = 0; j < element.spacedim(); j++)
	c[i][j] = 0.0;
    }
  }
}
//-----------------------------------------------------------------------------
Form::~Form()
{
  if ( c )
  {
    for (uint i = 0; i < ncoeff; i++)
      delete [] c[i];
    delete [] c;
  }
}
//-----------------------------------------------------------------------------
void Form::update(const Cell& cell)
{
  // Choose map from cell type
  switch ( cell.type() )
  {
  case Cell::triangle:
    updateTriLinMap(cell);
    break;
  case Cell::tetrahedron:
    updateTetLinMap(cell);
    break;
  default:
    dolfin_error("Unknown cell type.");
  }

  // Update coefficients
  updateCoefficients(cell);
}
//-----------------------------------------------------------------------------
void Form::updateTriLinMap(const Cell& cell)
{
  // Check that cell type is correct
  if ( cell.type() != Cell::triangle )
    dolfin_error("Wrong cell type for map (must be a triangle).");
  
  // Get coordinates
  const Point& p0(cell.coord(0));
  const Point& p1(cell.coord(1));
  const Point& p2(cell.coord(2));

  // Compute Jacobian of map
  f00 = p1.x - p0.x; f12 = p2.x - p0.x;
  f10 = p1.y - p0.y; f22 = p2.y - p0.y;
  
  // Compute determinant
  det = f00 * f11 - f01 * f10;
  
  // Check determinant
  if ( fabs(det) < DOLFIN_EPS )
    dolfin_error("Map from reference cell is singular.");
  
  // Compute inverse of Jacobian
  g00 =   f11 / det; g01 = - f01 / det;
  g10 = - f10 / det; g11 =   f00 / det;

  // Reset unused variables
  f02 = f12 = f20 = f21 = f22 = 0.0;
  g02 = g12 = g20 = g21 = g22 = 0.0;

  // Take absolute value of determinant
  det = fabs(det);
}
//-----------------------------------------------------------------------------
void Form::updateTetLinMap(const Cell& cell)
{
  // Check that cell type is correct
  if ( cell.type() != Cell::tetrahedron )
    dolfin_error("Wrong cell type for map (must be a tetrahedron).");
  
  // Get coordinates
  const Point& p0(cell.coord(0));
  const Point& p1(cell.coord(1));
  const Point& p2(cell.coord(2));
  const Point& p3(cell.coord(3));
  
  // Compute Jacobian of map
  f00 = p1.x - p0.x; f01 = p2.x - p0.x; f02 = p3.x - p0.x;
  f10 = p1.y - p0.y; f11 = p2.y - p0.y; f12 = p3.y - p0.y;
  f20 = p1.z - p0.z; f21 = p2.z - p0.z; f22 = p3.z - p0.z;
  
  // Compute sub-determinants
  real d00 = f11*f22 - f12*f21;
  real d01 = f12*f20 - f10*f22;
  real d02 = f10*f21 - f11*f20;
  
  real d10 = f02*f21 - f01*f22;
  real d11 = f00*f22 - f02*f20;
  real d12 = f01*f20 - f00*f21;
  
  real d20 = f01*f12 - f02*f11;
  real d21 = f02*f10 - f00*f12;
  real d22 = f00*f11 - f01*f10;
  
  // Compute determinant
  det = f00 * d00 + f10 * d10 + f20 * d20;
  
  // Check determinant
  if ( fabs(det) < DOLFIN_EPS )
    dolfin_error("Map from reference cell is singular.");
  
  // Compute inverse of Jacobian
  g00 = d00 / det; g01 = d10 / det; g02 = d20 / det;
  g10 = d01 / det; g11 = d11 / det; g12 = d21 / det;
  g20 = d02 / det; g21 = d12 / det; g22 = d22 / det;

  // Take absolute value of determinant
  det = fabs(det);
}
//-----------------------------------------------------------------------------
void Form::updateCoefficients(const Cell& cell)
{
  // Compute the projection of all functions to the current element
  for (uint i = 0; i < ncoeff; i++)
    functions[i]->project(cell, element, c[i]);
}
//-----------------------------------------------------------------------------
void Form::add(const NewFunction& function)
{
  functions.push_back(&function);
}
//-----------------------------------------------------------------------------
