// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-02-09
// Last changed: 2006-05-07

#include <dolfin/Vector.h>
#include <dolfin/P1tri.h>
#include <dolfin/FiniteElement.h>
#include <dolfin/ConstantFunction.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
ConstantFunction::ConstantFunction(real value)
  : GenericFunction(),
    value(value), _mesh(0), mesh_local(false)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
ConstantFunction::ConstantFunction(const ConstantFunction& f)
  : GenericFunction(),
    value(f.value), _mesh(f._mesh), mesh_local(false)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
ConstantFunction::~ConstantFunction()
{
  // Delete mesh if local
  if ( mesh_local )
    delete _mesh;
}
//-----------------------------------------------------------------------------
real ConstantFunction::operator()(const Point& p, uint i)
{
  return value;
}
//-----------------------------------------------------------------------------
real ConstantFunction::operator() (const Vertex& vertex, uint i)
{
  return value;
}
//-----------------------------------------------------------------------------
void ConstantFunction::sub(uint i)
{
  // Do nothing (value same for all components anyway)
}
//-----------------------------------------------------------------------------
void ConstantFunction::interpolate(real coefficients[],
				   AffineMap& map,
				   FiniteElement& element)
{
  // Evaluate function at interpolation points
  for (uint i = 0; i < element.spacedim(); i++)
    coefficients[i] = value;
}
//-----------------------------------------------------------------------------
dolfin::uint ConstantFunction::vectordim() const
{
  dolfin_error("Vector dimension unknown for constant function.");
  return 0;
}
//-----------------------------------------------------------------------------
#ifdef HAVE_PETSC_H
Vector& ConstantFunction::vector()
{
  dolfin_error("No vector associated with function (and none can be attached).");
  return *(new Vector()); // Code will not be reached, make compiler happy
}
#endif
//-----------------------------------------------------------------------------
Mesh& ConstantFunction::mesh()
{
  if ( !_mesh )
    dolfin_error("No mesh associated with function (try attaching one).");
  return *_mesh;
}
//-----------------------------------------------------------------------------
FiniteElement& ConstantFunction::element()
{
  dolfin_error("No finite element associated with function (an none can be attached).");
  return *(new P1tri()); // Code will not be reached, make compiler happy
}
//-----------------------------------------------------------------------------
#ifdef HAVE_PETSC_H
void ConstantFunction::attach(Vector& x, bool local)
{
  dolfin_error("Cannot attach vectors to constant functions.");
}
#endif
//-----------------------------------------------------------------------------
void ConstantFunction::attach(Mesh& mesh, bool local)
{
  // Delete old mesh if local
  if ( mesh_local )
    delete _mesh;

  // Attach new mesh
  _mesh = &mesh;
  mesh_local = local;
}
//-----------------------------------------------------------------------------
void ConstantFunction::attach(FiniteElement& element, bool local)
{
  dolfin_error("Cannot attach finite elements to constant functions.");
}
//-----------------------------------------------------------------------------
