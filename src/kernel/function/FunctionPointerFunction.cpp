// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-11-28
// Last changed: 2005-11-30

#include <dolfin/Node.h>
#include <dolfin/Vector.h>
#include <dolfin/P1Tri.h>
#include <dolfin/FiniteElement.h>
#include <dolfin/FunctionPointerFunction.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
FunctionPointerFunction::FunctionPointerFunction(FunctionPointer f,
						 uint vectordim)
  : f(f), _vectordim(vectordim), _mesh(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
FunctionPointerFunction::FunctionPointerFunction(const FunctionPointerFunction& f)
  : f(f.f), _vectordim(f._vectordim), _mesh(f._mesh)
{
  // Do nothing, just copy the values
}
//-----------------------------------------------------------------------------
FunctionPointerFunction::~FunctionPointerFunction()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
real FunctionPointerFunction::operator()(const Point& p, uint i)
{
  // Call overloaded evaluation operator at given point
  return (*f)(p, i);
}
//-----------------------------------------------------------------------------
real FunctionPointerFunction::operator() (const Node& node, uint i)
{
  // Call overloaded evaluation operator at given node
  return (*f)(node.coord(), i);
}
//-----------------------------------------------------------------------------
void FunctionPointerFunction::interpolate(real coefficients[],
					  AffineMap& map,
					  FiniteElement& element)
{
  // Initialize local data (if not already initialized correctly)
  local.init(element);
  
  // Map interpolation points to current cell
  element.pointmap(local.points, local.components, map);

  // Evaluate function at interpolation points
  for (uint i = 0; i < element.spacedim(); i++)
    coefficients[i] = (*f)(local.points[i], local.components[i]);
}
//-----------------------------------------------------------------------------
dolfin::uint FunctionPointerFunction::vectordim() const
{
  /// Return vector dimension of function
  return _vectordim;
}
//-----------------------------------------------------------------------------
Vector& FunctionPointerFunction::vector()
{
  dolfin_error("No vector associated with function (and none can be attached).");
  return *(new Vector()); // Code will not be reached, make compiler happy
}
//-----------------------------------------------------------------------------
Mesh& FunctionPointerFunction::mesh()
{
  if ( !_mesh )
    dolfin_error("No mesh associated with function (try attaching one).");
  return *_mesh;
}
//-----------------------------------------------------------------------------
FiniteElement& FunctionPointerFunction::element()
{
  dolfin_error("No finite element associated with function (an none can be attached).");
  return *(new P1Tri()); // Code will not be reached, make compiler happy
}
//-----------------------------------------------------------------------------
void FunctionPointerFunction::attach(Vector& x)
{
  dolfin_error("Cannot attach vectors to user-defined functions.");
}
//-----------------------------------------------------------------------------
void FunctionPointerFunction::attach(Mesh& mesh)
{
  _mesh = &mesh;
}
//-----------------------------------------------------------------------------
void FunctionPointerFunction::attach(FiniteElement& element)
{
  dolfin_error("Cannot attach finite elements to user-defined functions.");
}
//-----------------------------------------------------------------------------
