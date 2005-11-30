// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-11-26
// Last changed: 2005-11-29
//
// Note: this breaks the standard envelope-letter idiom slightly,
// since we call the envelope class from one of the letter classes.

#include <dolfin/Node.h>
#include <dolfin/Vector.h>
#include <dolfin/P1Tri.h>
#include <dolfin/FiniteElement.h>
#include <dolfin/Function.h>
#include <dolfin/UserFunction.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
UserFunction::UserFunction(Function* f, uint vectordim)
  : f(f), _vectordim(vectordim),  _mesh(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
UserFunction::~UserFunction()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
real UserFunction::operator()(const Point& p, uint i)
{
  dolfin_info("User-defined functions must implement real operator()(const Point& p, unsigned int i).");
  dolfin_error("Missing evaluation operator.");
  return 0.0;
}
//-----------------------------------------------------------------------------
real UserFunction::operator() (const Node& node, uint i)
{
  // Call overloaded evaluation operator at given node
  return (*f)(node.coord(), i);
}
//-----------------------------------------------------------------------------
void UserFunction::interpolate(real coefficients[], AffineMap& map,
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
dolfin::uint UserFunction::vectordim() const
{
  /// Return vector dimension of function
  return _vectordim;
}
//-----------------------------------------------------------------------------
Vector& UserFunction::vector()
{
  dolfin_error("No vector associated with function (and none can be attached).");
  return *(new Vector()); // Code will not be reached, make compiler happy
}
//-----------------------------------------------------------------------------
Mesh& UserFunction::mesh()
{
  if ( !_mesh )
    dolfin_error("No mesh associated with function (try attaching one).");
  return *_mesh;
}
//-----------------------------------------------------------------------------
FiniteElement& UserFunction::element()
{
  dolfin_error("No finite element associated with function (an none can be attached).");
  return *(new P1Tri()); // Code will not be reached, make compiler happy
}
//-----------------------------------------------------------------------------
void UserFunction::attach(Vector& x)
{
  dolfin_error("Cannot attach vectors to user-defined functions.");
}
//-----------------------------------------------------------------------------
void UserFunction::attach(Mesh& mesh)
{
  _mesh = &mesh;
}
//-----------------------------------------------------------------------------
void UserFunction::attach(FiniteElement& element)
{
  dolfin_error("Cannot attach finite elements to user-defined functions.");
}
//-----------------------------------------------------------------------------
