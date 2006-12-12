// Copyright (C) 2005-2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-11-28
// Last changed: 2006-12-12

#include <dolfin/Vertex.h>
#include <dolfin/Vector.h>
#include <dolfin/P1tri.h>
#include <dolfin/FiniteElement.h>
#include <dolfin/FunctionPointerFunction.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
FunctionPointerFunction::FunctionPointerFunction(FunctionPointer f,
						 uint vectordim)
  : GenericFunction(),
    f(f), _vectordim(vectordim), component(0), _mesh(0),
    mesh_local(false)
{
  // Check vector dimension
  if ( _vectordim == 0 )
    dolfin_error("Vector-valued function must have at least one component.");
}
//-----------------------------------------------------------------------------
FunctionPointerFunction::FunctionPointerFunction(const FunctionPointerFunction& f)
  : GenericFunction(),
    f(f.f), _vectordim(f._vectordim), component(0), _mesh(f._mesh),
    mesh_local(false)
{
  // Check vector dimension
  if ( _vectordim == 0 )
    dolfin_error("Vector-valued function must have at least one component.");
}
//-----------------------------------------------------------------------------
FunctionPointerFunction::~FunctionPointerFunction()
{
  // Delete mesh if local
  if ( mesh_local )
    delete _mesh;
}
//-----------------------------------------------------------------------------
real FunctionPointerFunction::operator()(const Point& p, uint i)
{
  // Call function at given point
  return (*f)(p, component + i);
}
//-----------------------------------------------------------------------------
real FunctionPointerFunction::operator() (const Vertex& vertex, uint i)
{
  // Call function at given vertex
  return (*f)(vertex.point(), component + i);
}
//-----------------------------------------------------------------------------
void FunctionPointerFunction::sub(uint i)
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
void FunctionPointerFunction::interpolate(real coefficients[], Cell& cell,
					  AffineMap& map, FiniteElement& element)
{
  // Initialize local data (if not already initialized correctly)
  local.init(element);
  
  // Map interpolation points to current cell
  element.pointmap(local.points, local.components, map);

  // Evaluate function at interpolation points
  for (uint i = 0; i < element.spacedim(); i++)
    coefficients[i] = (*f)(local.points[i], component + local.components[i]);
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
  return *(new P1tri()); // Code will not be reached, make compiler happy
}
//-----------------------------------------------------------------------------
void FunctionPointerFunction::attach(Vector& x, bool local)
{
  dolfin_error("Cannot attach vectors to user-defined functions.");
}
//-----------------------------------------------------------------------------
void FunctionPointerFunction::attach(Mesh& mesh, bool local)
{
  // Delete old mesh if local
  if ( mesh_local )
    delete _mesh;

  // Attach new mesh
  _mesh = &mesh;
  mesh_local = local;
}
//-----------------------------------------------------------------------------
void FunctionPointerFunction::attach(FiniteElement& element, bool local)
{
  dolfin_error("Cannot attach finite elements to user-defined functions.");
}
//-----------------------------------------------------------------------------
