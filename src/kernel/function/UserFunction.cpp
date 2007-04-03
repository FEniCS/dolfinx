// Copyright (C) 2005-2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-11-26
// Last changed: 2006-12-12
//
// Note: this breaks the standard envelope-letter idiom slightly,
// since we call the envelope class from one of the letter classes.

#include <dolfin/Vertex.h>
#include <dolfin/Vector.h>
#include <dolfin/P1tri.h>
#include <dolfin/FiniteElement.h>
#include <dolfin/Function.h>
#include <dolfin/UserFunction.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
UserFunction::UserFunction(Function* f, uint vectordim)
  : GenericFunction(),
    f(f), _vectordim(vectordim), component(0), _mesh(0),
    mesh_local(false)
{
  // Check vector dimension
  if ( _vectordim == 0 )
    dolfin_error("Vector-valued function must have at least one component.");
}
//-----------------------------------------------------------------------------
UserFunction::UserFunction(const UserFunction& f)
  : GenericFunction(),
    f(f.f), _vectordim(f._vectordim), component(0), _mesh(f._mesh),
    mesh_local(false)
{
  // Check vector dimension
  if ( _vectordim == 0 )
    dolfin_error("Vector-valued function must have at least one component.");
}
//-----------------------------------------------------------------------------
UserFunction::~UserFunction()
{
  // Delete mesh if local
  if ( mesh_local )
    delete _mesh;
}
//-----------------------------------------------------------------------------
real UserFunction::operator()(const Point& p, uint i)
{
  // Call overloaded eval function at given vertex
  return f->eval(p, component + i);
}
//-----------------------------------------------------------------------------
real UserFunction::operator() (const Vertex& vertex, uint i)
{
  // Call overloaded eval function at given vertex
  return f->eval(vertex.point(), component + i);
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
/*
void UserFunction::interpolate(real coefficients[],
                               const ufc::mesh& mesh,
                               const ufc::cell& cell,
                               const ufc::finite_element& finite_element) const
{
// Evaluate each dof to get coefficients for nodal basis expansion
  for (uint i = 0; i < finite_element.space_dimension(); i++)
    coefficients[i] = finite_element.evaluate_dof(i, *this, cell);
}
*/
//-----------------------------------------------------------------------------
void UserFunction::interpolate(real coefficients[], Cell& cell,
                               AffineMap& map, FiniteElement& element)
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
  return *(new P1tri()); // Code will not be reached, make compiler happy
}
//-----------------------------------------------------------------------------
void UserFunction::attach(Vector& x, bool local)
{
  dolfin_error("Cannot attach vectors to user-defined functions.");
}
//-----------------------------------------------------------------------------
void UserFunction::attach(Mesh& mesh, bool local)
{
  // Delete old mesh if local
  if ( mesh_local )
    delete _mesh;

  // Attach new mesh
  _mesh = &mesh;
  mesh_local = local;
}
//-----------------------------------------------------------------------------
void UserFunction::attach(FiniteElement& element, bool local)
{
  dolfin_error("Cannot attach finite elements to user-defined functions.");
}
//-----------------------------------------------------------------------------
void UserFunction::interpolate(real* coefficients,
                               const ufc::cell& cell,
                               const ufc::finite_element& finite_element)
{
  // Evaluate each dof to get coefficients for nodal basis expansion
  for (uint i = 0; i < finite_element.space_dimension(); i++)
    coefficients[i] = finite_element.evaluate_dof(i, *this, cell);
}
//-----------------------------------------------------------------------------
void UserFunction::evaluate(real* values,
                                const real* coordinates,
                                const ufc::cell& cell) const
{
  dolfin_error("Not implemented");
}
//-----------------------------------------------------------------------------
