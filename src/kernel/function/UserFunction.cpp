// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-11-26
// Last changed: 2005-11-30
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
  : GenericFunction(),
    f(f), _vectordim(vectordim), component(0), _mesh(0)
{
  // Check vector dimension
  if ( _vectordim == 0 )
    dolfin_error("Vector-valued function must have at least one component.");
}
//-----------------------------------------------------------------------------
UserFunction::UserFunction(const UserFunction& f)
  : GenericFunction(),
    f(f.f), _vectordim(f._vectordim), component(0), _mesh(f._mesh)
{
  // Check vector dimension
  if ( _vectordim == 0 )
    dolfin_error("Vector-valued function must have at least one component.");
}
//-----------------------------------------------------------------------------
UserFunction::~UserFunction()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
real UserFunction::operator()(const Point& p, uint i)
{
  // Call overloaded eval function at given node
  return f->eval(p, component + i);
}
//-----------------------------------------------------------------------------
real UserFunction::operator() (const Node& node, uint i)
{
  // Call overloaded eval function at given node
  return f->eval(node.coord(), component + i);
}
//-----------------------------------------------------------------------------
void UserFunction::sub(uint i)
{
  // Check if function is vector-valued
  if ( _vectordim == 1 )
    dolfin_error("Cannot pick component of scalar function.");

  // Check the dimension
  if ( i >= _vectordim )
    dolfin_error2("Illegal component index %d for function with %d components.",
		  i, _vectordim);

  // Save the component and make function scalar
  component = i;
  _vectordim = 1;
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
    coefficients[i] = f->eval(local.points[i], component + local.components[i]);
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
